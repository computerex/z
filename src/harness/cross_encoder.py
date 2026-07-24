"""Cross-encoder memory ranking — reranks memory candidates by relevance to current query."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Max memories to return
MAX_MEMORIES = 5


@dataclass
class MemoryCandidate:
    """A memory candidate with relevance score."""

    filepath: str
    description: str
    content_preview: str
    score: float = 0.0


class KeywordRanker:
    """Ranks memory files by relevance to a query using keyword overlap."""

    def rank(
        self,
        query: str,
        candidates: List[MemoryCandidate],
        top_k: int = MAX_MEMORIES,
    ) -> List[MemoryCandidate]:
        """Rank memory candidates by keyword overlap with the query.

        Args:
            query: The user's input text.
            candidates: List of memory candidates to rank.
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of top-k candidates (highest score first).
        """
        if not candidates:
            return []

        return self._rank_keyword(query, candidates, top_k)

    @staticmethod
    def _rank_keyword(
        query: str,
        candidates: List[MemoryCandidate],
        top_k: int,
    ) -> List[MemoryCandidate]:
        """Rank using keyword overlap (TF-weighted)."""
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
_ranker: Optional[KeywordRanker] = None


def get_ranker() -> KeywordRanker:
    """Get or create the global keyword ranker."""
    global _ranker
    if _ranker is None:
        _ranker = KeywordRanker()
    return _ranker


def rank_memories(
    query: str,
    candidates: List[MemoryCandidate],
    top_k: int = MAX_MEMORIES,
) -> List[MemoryCandidate]:
    """Convenience function to rank memories."""
    return get_ranker().rank(query, candidates, top_k)
