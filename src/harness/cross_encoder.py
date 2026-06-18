"""
Local cross-encoder for memory relevance ranking.

Uses sentence-transformers cross-encoder models to rank memory files
by relevance to a user query. Falls back to simple keyword overlap
when the model is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default model — small, fast, good for relevance scoring
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Fallback model if default is too large
FALLBACK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
# Max memories to return
MAX_MEMORIES = 5
# Min score to include (0-1, cross-encoder produces logits not normalized)
# We use a softmin approach instead


@dataclass
class MemoryCandidate:
    """A memory candidate with relevance score."""

    filepath: str
    description: str
    content_preview: str
    score: float = 0.0


class CrossEncoderRanker:
    """Ranks memory files by relevance to a query using a local cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        """Check if the cross-encoder model is available."""
        if self._model is not None:
            return True
        try:
            self._lazy_load()
            return self._model is not None
        except Exception:
            return False

    def _lazy_load(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        for candidate in [self.model_name, FALLBACK_MODEL]:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(candidate)
                self.model_name = candidate
                logger.debug(
                    "Loaded cross-encoder model: %s", candidate
                )
                return
            except Exception as e:
                logger.debug(
                    "Failed to load cross-encoder %s: %s", candidate, e
                )
                continue

        logger.warning(
            "No cross-encoder model available — will use keyword fallback"
        )

    def rank(
        self,
        query: str,
        candidates: List[MemoryCandidate],
        top_k: int = MAX_MEMORIES,
    ) -> List[MemoryCandidate]:
        """Rank memory candidates by relevance to the query.

        Args:
            query: The user's input text.
            candidates: List of memory candidates to rank.
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of top-k candidates (highest score first).
        """
        if not candidates:
            return []

        self._lazy_load()

        if self._model is not None:
            return self._rank_with_model(query, candidates, top_k)
        else:
            return self._rank_keyword(query, candidates, top_k)

    def _rank_with_model(
        self,
        query: str,
        candidates: List[MemoryCandidate],
        top_k: int,
    ) -> List[MemoryCandidate]:
        """Rank using cross-encoder model."""
        # Build pairs: (query, memory_description + content_preview)
        pairs = [
            (query, f"{c.description}: {c.content_preview}"[:512])
            for c in candidates
        ]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)

            for i, candidate in enumerate(candidates):
                # scores are logits — normalize via sigmoid-ish scaling
                candidate.score = float(scores[i])

            candidates.sort(key=lambda c: c.score, reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.debug("Cross-encoder prediction failed: %s", e)
            return self._rank_keyword(query, candidates, top_k)

    def _rank_keyword(
        self,
        query: str,
        candidates: List[MemoryCandidate],
        top_k: int,
    ) -> List[MemoryCandidate]:
        """Fallback ranking using keyword overlap (TF-weighted)."""
        # Tokenize query into lowercase words, filter stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "about",
            "up", "and", "but", "or", "if", "what", "which", "who", "whom",
            "this", "that", "these", "those", "it", "its", "i", "me", "my",
            "we", "our", "you", "your", "he", "she", "they", "them", "their",
        }
        query_words = set(
            w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower())
            if w not in stopwords and len(w) > 2
        )

        if not query_words:
            # If query has no meaningful words, return as-is
            return candidates[:top_k]

        for candidate in candidates:
            text = f"{candidate.description} {candidate.content_preview}".lower()
            text_words = set(re.findall(r"[a-zA-Z0-9_]+", text))
            if not text_words:
                candidate.score = 0.0
                continue
            # Jaccard-like overlap score
            overlap = len(query_words & text_words)
            # TF bonus: count occurrences
            tf_score = sum(
                text.count(w) for w in query_words if w in text
            )
            candidate.score = overlap + (tf_score * 0.1)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return [c for c in candidates[:top_k] if c.score > 0]


# Module-level singleton with lazy init
_ranker: Optional[CrossEncoderRanker] = None


def get_ranker(model_name: str = DEFAULT_MODEL) -> CrossEncoderRanker:
    """Get or create the global cross-encoder ranker."""
    global _ranker
    if _ranker is None:
        _ranker = CrossEncoderRanker(model_name)
    return _ranker


def rank_memories(
    query: str,
    candidates: List[MemoryCandidate],
    top_k: int = MAX_MEMORIES,
) -> List[MemoryCandidate]:
    """Convenience function to rank memories."""
    return get_ranker().rank(query, candidates, top_k)
