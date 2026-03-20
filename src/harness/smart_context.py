"""Smart context management for long-running autonomous agent sessions.

Designed around these principles:

1. ALL message types are compaction candidates — assistant reasoning,
   tool results, file reads, command output. Not just tool results.
2. Relevance is computed via semantic embedding similarity (all-MiniLM-L6-v2,
   22M params) between messages and active todo context.  Falls back to
   keyword matching if the model can't load.
3. Proactive compaction triggers before hitting the hard context limit,
   creating headroom so the system isn't always in crisis mode.
4. Compacted content leaves breadcrumb traces so the agent can recover
   by re-reading files or re-running commands.
5. System prompt (index 0) and first user-assistant pair (indices 1-2)
   are never touched — they anchor the entire conversation.
"""

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Set, TYPE_CHECKING
from collections import Counter

import time as _time_mod_sc

_sc_t0 = _time_mod_sc.perf_counter()
import numpy as np

_sc_t1 = _time_mod_sc.perf_counter()

from .context_management import estimate_tokens, estimate_messages_tokens
from .logger import get_logger


_sc_t2 = _time_mod_sc.perf_counter()

import logging as _logging_sc

_logging_sc.getLogger("harness.smart_context.boot").info(
    "smart_context import: numpy=%.0fms sklearn=disabled total=%.0fms",
    (_sc_t1 - _sc_t0) * 1000,
    (_sc_t2 - _sc_t0) * 1000,
)

if TYPE_CHECKING:
    from .todo_manager import TodoManager


log = get_logger("smart_context")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Message indices that must NEVER be compacted or consolidated.
# 0 = system prompt, 1 = first user message, 2 = first assistant response.
PROTECTED_INDICES = frozenset({0, 1, 2})

# Minimum token count for a message to be worth compacting.
MIN_COMPACT_TOKENS = 80

# Unicode marker identifying already-compacted messages.
COMPACT_MARKER = "\u25c7"  # ◇

# Maximum characters to embed per message (keeps embedding fast + focused).
_EMBED_MAX_CHARS = 2000

# Assistant messages above this size get excerpted (start/middle/end) and
# stored for later retrieval, similar to compacted tool results.
_LONG_ASSISTANT_COMPACT_CHARS = 3000


# ---------------------------------------------------------------------------
# SemanticScorer — lazy-loaded embedding model for relevance scoring
# ---------------------------------------------------------------------------


class SemanticScorer:
    """Computes semantic similarity using a small bi-encoder embedding model.

    Uses ``all-MiniLM-L6-v2`` (22M params, 384-dim, <10ms/embed on CPU).
    Embeddings are cached by content hash so each message is encoded once.

    Falls back to keyword matching if the model cannot load (e.g. missing
    dependency, offline, PyInstaller without model bundled).
    """

    _instance: Optional["SemanticScorer"] = None

    def __init__(self):
        self._model = None
        self._load_failed = False
        self._cache: Dict[str, np.ndarray] = {}  # content_hash -> embedding
        self._max_cache = 500

    @classmethod
    def get(cls) -> "SemanticScorer":
        """Singleton accessor — avoids loading the model multiple times."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- lazy load ----------------------------------------------------------

    def _ensure_model(self) -> bool:
        """Load the model on first use. Returns True if model is available."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            _t0_import = time.perf_counter()
            from sentence_transformers import SentenceTransformer

            _t1_import = time.perf_counter()

            # Some environments inject a broken loopback proxy (127.0.0.1:9),
            # which blocks model fetch even when network is otherwise available.
            # Temporarily clear only known-bad proxy values while loading.
            proxy_keys = (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            )
            old_proxy_vals: Dict[str, Optional[str]] = {}
            try:
                for k in proxy_keys:
                    v = os.environ.get(k)
                    old_proxy_vals[k] = v
                    if v and ("127.0.0.1:9" in v or "localhost:9" in v):
                        os.environ.pop(k, None)

                # Only use locally cached model - NEVER download automatically
                # Downloading can hang indefinitely with no timeout and blocks Ctrl+C
                self._model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    local_files_only=True,
                )
                # If we get here, model loaded successfully from cache
            finally:
                for k, v in old_proxy_vals.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            _t2_load = time.perf_counter()
            log.info(
                "SemanticScorer model ready: import=%.0fms load=%.0fms total=%.0fms",
                (_t1_import - _t0_import) * 1000,
                (_t2_load - _t1_import) * 1000,
                (_t2_load - _t0_import) * 1000,
            )
            return True
        except Exception:
            self._load_failed = True
            return False

    @property
    def available(self) -> bool:
        return self._ensure_model()

    # -- embedding ----------------------------------------------------------

    def _content_key(self, text: str) -> str:
        """Fast hash for cache lookup."""
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

    def _embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, using cache."""
        # Truncate for speed + to keep embedding focused on salient content
        truncated = text[:_EMBED_MAX_CHARS]
        key = self._content_key(truncated)

        if key in self._cache:
            return self._cache[key]

        vec = self._model.encode(truncated, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)

        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_cache:
            # Remove the first 100 entries (roughly LRU-ish)
            to_remove = list(self._cache.keys())[:100]
            for k in to_remove:
                del self._cache[k]

        self._cache[key] = vec
        return vec

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently, using cache where possible."""
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        to_encode: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            truncated = text[:_EMBED_MAX_CHARS]
            key = self._content_key(truncated)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_encode.append((i, truncated))

        if to_encode:
            batch_texts = [t for _, t in to_encode]
            vecs = self._model.encode(
                batch_texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            for (idx, truncated), vec in zip(to_encode, vecs):
                vec = np.asarray(vec, dtype=np.float32)
                key = self._content_key(truncated)
                self._cache[key] = vec
                results[idx] = vec

        # Evict if cache grew too large
        if len(self._cache) >= self._max_cache:
            to_remove = list(self._cache.keys())[:100]
            for k in to_remove:
                del self._cache[k]

        return results  # type: ignore[return-value]

    # -- scoring ------------------------------------------------------------

    def score_relevance(self, content: str, query: str) -> float:
        """Compute semantic similarity between content and query.

        Returns 0.0 (unrelated) to 1.0 (identical meaning).
        Falls back to keyword heuristic if model unavailable.
        """
        if not self._ensure_model():
            return self._keyword_fallback(content, query)

        try:
            vec_content = self._embed_one(content)
            vec_query = self._embed_one(query)
            # Cosine similarity (vectors are already normalized)
            sim = float(np.dot(vec_content, vec_query))
            # Clamp to [0, 1] — cosine can be slightly negative
            return max(0.0, min(1.0, sim))
        except Exception:
            return self._keyword_fallback(content, query)

    def score_batch(
        self,
        contents: List[str],
        query: str,
    ) -> List[float]:
        """Score multiple contents against a single query. Much faster than
        calling ``score_relevance`` in a loop because it batch-encodes.
        """
        if not self._ensure_model() or not contents:
            return [self._keyword_fallback(c, query) for c in contents]

        try:
            vec_query = self._embed_one(query)
            vecs = self._embed_batch(contents)
            scores = []
            for v in vecs:
                sim = float(np.dot(v, vec_query))
                scores.append(max(0.0, min(1.0, sim)))
            return scores
        except Exception:
            return [self._keyword_fallback(c, query) for c in contents]

    @staticmethod
    def _keyword_fallback(content: str, query: str) -> float:
        """Simple keyword overlap fallback when embedding is unavailable."""
        if not query:
            return 0.4  # neutral
        query_words = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", query.lower()))
        noise = {
            "the",
            "and",
            "for",
            "this",
            "that",
            "with",
            "from",
            "are",
            "was",
            "has",
            "have",
            "not",
            "but",
            "can",
            "all",
            "will",
            "use",
            "add",
            "into",
            "also",
            "any",
            "each",
            "get",
            "set",
            "should",
            "would",
            "need",
            "make",
            "like",
            "new",
            "see",
            "now",
            "just",
            "its",
            "our",
            "file",
            "code",
            "test",
            "run",
            "check",
            "fix",
            "error",
            "result",
        }
        query_words -= noise
        if not query_words:
            return 0.4
        content_lower = content[:3000].lower()
        hits = sum(1 for w in query_words if w in content_lower)
        return min(0.9, hits / max(1, len(query_words) * 0.3))


# ---------------------------------------------------------------------------
# Compaction trace — breadcrumb left when content is compacted
# ---------------------------------------------------------------------------


@dataclass
class CompactionTrace:
    """Breadcrumb left when content is compacted, enabling recovery."""

    original_type: (
        str  # 'file_read', 'command_output', 'search_result', 'assistant', ...
    )
    source: str  # file path, command string, or brief label
    summary: str  # one-line description of what was there
    tokens_freed: int = 0
    compacted_at: float = field(default_factory=time.time)

    def format_notice(self) -> str:
        """Format as an inline notice the agent can act on."""
        if self.original_type == "file_read":
            return (
                f"[{COMPACT_MARKER} File '{self.source}' was read here "
                f"({self.tokens_freed} tok freed). Re-read if needed.]"
            )
        elif self.original_type == "duplicate_read":
            return (
                f"[{COMPACT_MARKER} Duplicate read of '{self.source}' — "
                f"latest version exists later in conversation.]"
            )
        elif self.original_type == "command_output":
            return (
                f"[{COMPACT_MARKER} Output of '{self.source}' was here "
                f"({self.tokens_freed} tok freed). Re-run if needed.]"
            )
        elif self.original_type == "search_result":
            return (
                f"[{COMPACT_MARKER} Search results were here "
                f"({self.tokens_freed} tok freed). Re-search if needed.]"
            )
        elif self.original_type == "assistant":
            return (
                f"[{COMPACT_MARKER} Previous analysis: {self.summary} "
                f"({self.tokens_freed} tok freed)]"
            )
        else:
            return (
                f"[{COMPACT_MARKER} {self.original_type}: {self.summary} "
                f"({self.tokens_freed} tok freed)]"
            )


# ---------------------------------------------------------------------------
# Context node metadata (persistent semantic breadcrumbs)
# ---------------------------------------------------------------------------


@dataclass
class ContextNodeMeta:
    """Persistent metadata for a message fingerprint across turns."""

    node_id: str
    first_seen: float
    last_seen: float
    seen_count: int
    role: str
    msg_type: str
    source: str
    last_tokens: int
    lifecycle_state: str = "warm"  # hot/warm/cold/archived


# ---------------------------------------------------------------------------
# Tool result storage — for retrieving compacted tool results
# ---------------------------------------------------------------------------


@dataclass
class StoredToolResult:
    """A stored tool result that can be retrieved after compaction."""

    result_id: str
    tool_name: str
    original_content: str
    timestamp: float
    message_index: int
    tokens: int


class ToolResultStorage:
    """Storage for tool results that have been compacted.

    Allows the agent to retrieve full tool results after they've been
    abbreviated in the conversation context.
    """

    # Maximum total bytes to store (100MB)
    MAX_TOTAL_BYTES = 100 * 1024 * 1024

    # Maximum age for results (1 hour)
    MAX_AGE_SECONDS = 3600

    def __init__(self, max_results: int = 100):
        self._results: Dict[str, StoredToolResult] = {}
        self._max_results = max_results
        self._access_order: List[str] = []  # For LRU eviction
        self._total_bytes = 0  # Track total bytes stored
        self._counter = 0  # For unique result IDs

    def _evict_if_needed(self) -> None:
        """Evict old results if we exceed limits."""
        # Evict by count
        while len(self._results) > self._max_results:
            if not self._access_order:
                break
            rid = self._access_order.pop(0)
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

        # Evict by size
        while self._total_bytes > self.MAX_TOTAL_BYTES and self._access_order:
            rid = self._access_order.pop(0)
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

    def _cleanup_old_results(self) -> int:
        """Clean up results older than MAX_AGE_SECONDS."""
        now = time.time()
        to_remove = [
            rid
            for rid, result in self._results.items()
            if now - result.timestamp > self.MAX_AGE_SECONDS
        ]
        if not to_remove:
            return 0

        # Use set for O(1) lookup instead of O(n) list.remove()
        to_remove_set = set(to_remove)

        # Filter access_order in O(n) instead of O(n²)
        self._access_order = [
            rid for rid in self._access_order if rid not in to_remove_set
        ]

        # Remove from results dict and update byte count
        for rid in to_remove:
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

        return len(to_remove)

    def store_result(
        self, tool_name: str, content: str, message_index: int = -1
    ) -> str:
        """Store a tool result and return its result_id.

        Args:
            tool_name: Name of the tool that produced this result
            content: Full content of the tool result
            message_index: Index of the message in the conversation (optional, not used after compaction)

        Returns:
            result_id: Unique identifier for this stored result
        """
        # Validate content size
        content_bytes = len(content.encode("utf-8", errors="replace"))
        if content_bytes > 10 * 1024 * 1024:  # 10MB limit per result
            raise ValueError(f"Tool result too large to store: {content_bytes:,} bytes")

        # Generate a unique result ID with counter to prevent collisions
        content_hash = hashlib.md5(
            content.encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        self._counter += 1
        result_id = f"res_{content_hash}_{self._counter}"

        result = StoredToolResult(
            result_id=result_id,
            tool_name=tool_name,
            original_content=content,
            timestamp=time.time(),
            message_index=message_index,
            tokens=estimate_tokens(content),
        )

        self._results[result_id] = result
        self._access_order.append(result_id)
        self._total_bytes += content_bytes

        # Clean up old results and evict if needed
        self._cleanup_old_results()
        self._evict_if_needed()

        return result_id

    def get_result(self, result_id: str) -> Optional[StoredToolResult]:
        """Retrieve a stored tool result by ID.

        Args:
            result_id: The ID of the stored result (format: res_<hash>_<counter>)

        Returns:
            StoredToolResult if found, None otherwise
        """
        # Validate result_id format to prevent security issues
        if not result_id or not isinstance(result_id, str):
            return None
        # Expected format: res_<8_hex_chars>_<counter>
        if not re.match(r"^res_[a-f0-9]{8}_\d+$", result_id):
            return None

        result = self._results.get(result_id)
        if result:
            # Update access order for LRU
            if result_id in self._access_order:
                self._access_order.remove(result_id)
            self._access_order.append(result_id)
        return result

    def list_results(self, tool_name: Optional[str] = None) -> List[StoredToolResult]:
        """List all stored results, optionally filtered by tool name.

        Args:
            tool_name: If provided, only return results from this tool

        Returns:
            List of stored results, ordered by most recently accessed
        """
        results = list(self._results.values())
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        # Sort by access order (most recent first)
        result_order = {rid: i for i, rid in enumerate(reversed(self._access_order))}
        results.sort(key=lambda r: result_order.get(r.result_id, float("inf")))
        return results

    def clear_old_results(self, max_age_seconds: float = 3600) -> int:
        """Clear results older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds (default: 1 hour)

        Returns:
            Number of results cleared
        """
        now = time.time()
        to_remove = [
            rid
            for rid, result in self._results.items()
            if now - result.timestamp > max_age_seconds
        ]
        for rid in to_remove:
            self._results.pop(rid, None)
            if rid in self._access_order:
                self._access_order.remove(rid)
        return len(to_remove)


# ---------------------------------------------------------------------------
# SmartContextManager
# ---------------------------------------------------------------------------


class SmartContextManager:
    """Manages context by scoring and compacting messages to stay within budget.

    Scoring heuristic (no ML — pure deterministic):
    - recency   (30%): newer messages score higher
    - relevance (35%): messages referencing files/keywords in active todos
    - regen     (35%): how hard is it to regenerate this content?
                       file reads are cheap (just re-read), assistant
                       reasoning is expensive (non-deterministic)
    - size pressure:   bigger messages are better compaction targets

    Messages are scored and compacted lowest-score-first until context
    fits within budget.  Duplicate file reads are consolidated first
    (always safe, big wins).

    Protected: system prompt (0), first user-assistant pair (1, 2),
    and the most recent N messages (``recent_window``).

    """

    def __init__(self, todo_manager: "TodoManager"):
        self.todo_manager = todo_manager
        self.compaction_traces: List[CompactionTrace] = []
        self._max_traces = 50
        self._scorer = SemanticScorer.get()

        # Storage for tool results that have been compacted
        self.result_storage = ToolResultStorage(max_results=100)

        # Budget: target this fraction of max_tokens after compaction.
        # E.g. 0.75 means we try to get context down to 75% of the hard
        # limit, creating 25% headroom so we don't compact again immediately.
        self.budget_ratio = 0.75

        # How many recent messages (from the tail) to always keep intact
        # for scoring/compaction (Phase 2).  Phase 3 eviction can still
        # remove them if they push us over budget.
        self.recent_window = 6

        # Maximum tokens to protect in the recent window.
        # If the last N messages exceed this, the oldest ones in the
        # window become eviction candidates.
        self.recent_window_max_tokens = 16_000
        # Automatic semantic-maintenance target for per-turn cleanup.
        self.soft_target_ratio = 0.62
        # Base cap for guidance pruning each tick; increases dynamically
        # when context utilization is already moderate/high.
        self.max_guidance_prune_per_tick = 8
        self.guidance_prune_boost_threshold = 0.55
        self.max_guidance_prune_per_tick_boosted = 12
        # Persistent semantic metadata keyed by message fingerprint.
        self.node_metadata: Dict[str, ContextNodeMeta] = {}
        # Telemetry snapshot for latest learned-prior scoring pass.
        self.last_policy_stats: Dict[str, Any] = {}

    # -- Backward-compat shims for code that references the old API --------

    @property
    def eviction_traces(self) -> List[CompactionTrace]:
        """Alias so old code referencing eviction_traces still works."""
        return self.compaction_traces

    @eviction_traces.setter
    def eviction_traces(self, value: List[CompactionTrace]):
        self.compaction_traces = value

    @property
    def protected_indices(self) -> frozenset:
        """Alias so old code/tests referencing protected_indices still works."""
        return PROTECTED_INDICES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact_context(
        self,
        messages: List[Any],
        max_tokens: int,
        current_tokens: Optional[int] = None,
        **_kwargs,  # absorb old 'aggressive' param
    ) -> Tuple[List[Any], int, str]:
        """Score every message, compact lowest-priority ones until under budget.

        Returns ``(messages, tokens_freed, report_text)``.
        """
        if current_tokens is None:
            current_tokens = estimate_messages_tokens(messages)

        budget = int(max_tokens * self.budget_ratio)
        debug_scoring = os.environ.get("HARNESS_DEBUG_SMART_CONTEXT", "0") == "1"

        log.debug(
            "compact_context start: current=%d budget=%d ratio=%.2f max=%d msgs=%d",
            current_tokens,
            budget,
            self.budget_ratio,
            max_tokens,
            len(messages),
        )

        if current_tokens <= budget:
            if debug_scoring:
                scored_preview = self._score_all_messages(messages, set(), set())
                scored_preview.sort(key=lambda x: x[1])
                preview_rows = []
                for index, score, msg_type, source in scored_preview[:8]:
                    tok = estimate_tokens(_get_content(messages[index]))
                    preview_rows.append(
                        f"idx={index} score={score:.3f} tok={tok} type={msg_type} src={source[:60]}"
                    )
                if preview_rows:
                    log.debug(
                        "compact_context skipped (under budget). lowest-score candidates: %s",
                        " | ".join(preview_rows),
                    )
            return messages, 0, ""

        excess = current_tokens - budget
        report_parts: List[str] = []
        total_freed = 0

        # Phase 1: consolidate duplicate file reads (always safe, big wins)
        messages, dup_freed = self._consolidate_duplicates(messages)
        if dup_freed > 0:
            total_freed += dup_freed
            report_parts.append(f"Deduplicated file reads: -{dup_freed:,} tok")

        if total_freed >= excess:
            return messages, total_freed, "; ".join(report_parts)

        # Phase 2: score every non-protected message, compact lowest first
        scored = self._score_all_messages(messages, set(), set())
        scored.sort(key=lambda x: x[1])  # lowest keep-priority first

        if debug_scoring and scored:
            preview_rows = []
            for index, score, msg_type, source in scored[:10]:
                tok = estimate_tokens(_get_content(messages[index]))
                preview_rows.append(
                    f"idx={index} score={score:.3f} tok={tok} type={msg_type} src={source[:60]}"
                )
            log.debug("scored candidates (lowest first): %s", " | ".join(preview_rows))

        tokens_still_needed = excess - total_freed
        compacted_count = 0

        for index, _score, msg_type, source in scored:
            if tokens_still_needed <= 0:
                break

            msg = messages[index]
            content = _get_content(msg)
            old_tokens = estimate_tokens(content)

            notice, summary = self._compact_message(content, msg_type, source, index)
            new_tokens = estimate_tokens(notice)
            freed = old_tokens - new_tokens
            if freed <= 0:
                continue

            _set_content(msg, notice)
            tokens_still_needed -= freed
            total_freed += freed
            compacted_count += 1

            trace = CompactionTrace(
                original_type=msg_type,
                source=source,
                summary=summary,
                tokens_freed=freed,
            )
            self.compaction_traces.append(trace)

        # Trim trace history
        if len(self.compaction_traces) > self._max_traces:
            self.compaction_traces = self.compaction_traces[-self._max_traces :]

        if compacted_count > 0:
            compact_freed = total_freed - dup_freed
            report_parts.append(
                f"Compacted {compacted_count} messages: -{compact_freed:,} tok"
            )
            if debug_scoring:
                type_counts = Counter(
                    t.original_type for t in self.compaction_traces[-compacted_count:]
                )
                log.debug(
                    "compacted message types: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
                )

        # Phase 2.5: Emergency compaction of oversized recent-window messages.
        # If Phase 2 couldn't find candidates (e.g. short conversation where
        # everything falls within the recent window), compact the largest
        # non-protected messages in-place without evicting them.
        current_tokens = estimate_messages_tokens(messages)
        if current_tokens > budget and compacted_count == 0:
            protected_recent = self._get_protected_recent(messages)
            emergency_candidates = []
            for i, msg in enumerate(messages):
                if i in PROTECTED_INDICES:
                    continue
                content = _get_content(msg)
                if not content or COMPACT_MARKER in content[:20]:
                    continue
                tokens = estimate_tokens(content)
                if tokens < MIN_COMPACT_TOKENS:
                    continue
                role = msg.role if hasattr(msg, "role") else msg.get("role", "")
                msg_type, source = self._classify(content, role)
                emergency_candidates.append((i, tokens, msg_type, source, content))

            # Compact largest messages first
            emergency_candidates.sort(key=lambda x: x[1], reverse=True)
            for i, tokens, msg_type, source, content in emergency_candidates:
                if current_tokens <= budget:
                    break
                notice, summary = self._compact_message(content, msg_type, source, i)
                new_tokens = estimate_tokens(notice)
                freed = tokens - new_tokens
                if freed <= 0:
                    continue
                _set_content(messages[i], notice)
                current_tokens -= freed
                total_freed += freed
                compacted_count += 1
                self.compaction_traces.append(
                    CompactionTrace(
                        original_type=msg_type,
                        source=source,
                        summary=summary,
                        tokens_freed=freed,
                    )
                )
            if compacted_count > 0:
                report_parts.append(
                    f"Emergency compacted {compacted_count} oversized messages"
                )

        # Phase 3: If STILL over budget after compaction, evict entire
        # lowest-scored messages.  Already-compacted messages (◇ marker)
        # are evicted first since their traces are already recorded,
        # then uncompacted messages in ascending score order.
        current_tokens = estimate_messages_tokens(messages)
        if current_tokens > budget:
            total_msg = len(messages)
            protected_recent = self._get_protected_recent(messages)

            # Bucket 1: already-compacted messages (cheap to drop, trace exists)
            compacted_indices: List[int] = []
            # Bucket 2: scored messages, lowest score first
            scored_evict = self._score_all_messages(messages, set(), set())
            scored_evict.sort(key=lambda x: x[1])  # lowest first
            scored_indices = {s[0] for s in scored_evict}

            for i, msg in enumerate(messages):
                if i in PROTECTED_INDICES or i in protected_recent:
                    continue
                content = _get_content(msg)
                if not content:
                    continue
                if COMPACT_MARKER in content[:20]:
                    compacted_indices.append(i)

            # Build ordered eviction list: compacted first, then scored
            eviction_order: List[int] = compacted_indices[:]
            for index, _score, _mt, _src in scored_evict:
                if index not in set(compacted_indices):
                    eviction_order.append(index)

            evicted_indices: set = set()
            for index in eviction_order:
                if current_tokens <= budget:
                    break
                msg = messages[index]
                content = _get_content(msg)
                freed = estimate_tokens(content)
                if freed <= 0:
                    continue
                evicted_indices.add(index)
                current_tokens -= freed
                total_freed += freed

                # Record a trace only for messages not already compacted
                if COMPACT_MARKER not in content[:20]:
                    role = msg.role if hasattr(msg, "role") else msg.get("role", "")
                    msg_type, source = self._classify(content, role)
                    summary = content.splitlines()[0][:100] if content else ""
                    self.compaction_traces.append(
                        CompactionTrace(
                            original_type=msg_type,
                            source=source,
                            summary=summary,
                            tokens_freed=freed,
                        )
                    )

            if evicted_indices:
                messages = [
                    m for i, m in enumerate(messages) if i not in evicted_indices
                ]
                report_parts.append(
                    f"Evicted {len(evicted_indices)} low-priority messages"
                )
                if debug_scoring:
                    log.debug("evicted indices: %s", sorted(evicted_indices))

        # Trim trace history
        if len(self.compaction_traces) > self._max_traces:
            self.compaction_traces = self.compaction_traces[-self._max_traces :]

        final_tokens = estimate_messages_tokens(messages)
        log.debug(
            "compact_context done: freed=%d final_tokens=%d report=%s",
            total_freed,
            final_tokens,
            "; ".join(report_parts),
        )
        return messages, total_freed, "; ".join(report_parts)

    def semantic_maintenance_tick(
        self,
        messages: List[Any],
        max_tokens: int,
        current_tokens: Optional[int] = None,
    ) -> Tuple[List[Any], int, str]:
        """Continuous low-overhead semantic maintenance.

        Runs each turn to keep context healthy even before hard limits.
        """
        if current_tokens is None:
            current_tokens = estimate_messages_tokens(messages)
        budget = int(max_tokens * self.soft_target_ratio)

        total_freed = 0
        report_parts: List[str] = []

        # Refresh semantic breadcrumbs each turn before maintenance actions.
        self._refresh_node_metadata(messages)

        utilization = (current_tokens / max_tokens) if max_tokens > 0 else 0.0
        guidance_limit = self.max_guidance_prune_per_tick
        if utilization >= self.guidance_prune_boost_threshold:
            guidance_limit = self.max_guidance_prune_per_tick_boosted

        # 1) Lifecycle-expire stale guidance/nudges first.
        messages, guidance_freed, guidance_count = self._compact_stale_guidance(
            messages, limit=guidance_limit
        )
        if guidance_freed > 0:
            total_freed += guidance_freed
            current_tokens = max(0, current_tokens - guidance_freed)
            report_parts.append(
                f"Guidance lifecycle pruning: {guidance_count} msgs, -{guidance_freed:,} tok"
            )

        # 2) If still above soft budget, run full compaction toward soft target.
        if current_tokens > budget:
            old_ratio = self.budget_ratio
            try:
                self.budget_ratio = self.soft_target_ratio
                messages, freed, report = self.compact_context(
                    messages, max_tokens, current_tokens=current_tokens
                )
                if freed > 0:
                    total_freed += freed
                    if report:
                        report_parts.append(report)
            finally:
                self.budget_ratio = old_ratio

        return messages, total_freed, "; ".join(report_parts)

    def _refresh_node_metadata(self, messages: List[Any]) -> None:
        """Update persistent metadata for each visible message."""
        now = time.time()
        total = len(messages)
        for i, msg in enumerate(messages):
            content = _get_content(msg)
            if not content:
                continue
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            msg_type, source = self._classify(content, role)
            node_id = _message_fingerprint(role, content)
            tokens = estimate_tokens(content)
            age_from_end = max(0, total - 1 - i)
            lifecycle = self._lifecycle_for_age(age_from_end)
            if COMPACT_MARKER in content[:20]:
                lifecycle = "archived"

            existing = self.node_metadata.get(node_id)
            if existing is None:
                self.node_metadata[node_id] = ContextNodeMeta(
                    node_id=node_id,
                    first_seen=now,
                    last_seen=now,
                    seen_count=1,
                    role=role,
                    msg_type=msg_type,
                    source=source,
                    last_tokens=tokens,
                    lifecycle_state=lifecycle,
                )
            else:
                existing.last_seen = now
                existing.seen_count += 1
                existing.role = role
                existing.msg_type = msg_type
                existing.source = source
                existing.last_tokens = tokens
                existing.lifecycle_state = lifecycle

        # Prevent unbounded metadata growth.
        if len(self.node_metadata) > 5000:
            items = sorted(self.node_metadata.values(), key=lambda x: x.last_seen)
            for meta in items[:1000]:
                self.node_metadata.pop(meta.node_id, None)

    @staticmethod
    def _lifecycle_for_age(age_from_end: int) -> str:
        if age_from_end <= 2:
            return "hot"
        if age_from_end <= 12:
            return "warm"
        return "cold"

    def build_context_recovery_notice(self) -> str:
        """Build a reorientation notice after context compaction/truncation.

        Summarises what was compacted and shows current todo state so the
        agent can decide whether to re-read any files.
        """
        if not self.compaction_traces:
            return ""

        recent = self.compaction_traces[-10:]

        # Group by type
        by_type: Dict[str, List[CompactionTrace]] = {}
        for t in recent:
            by_type.setdefault(t.original_type, []).append(t)

        lines = [
            "[CONTEXT COMPACTION NOTICE]",
            f"Some content was compacted to stay within context limits. "
            f"Total compactions this session: {len(self.compaction_traces)}",
            "",
        ]

        for typ, traces in by_type.items():
            sources = list(dict.fromkeys(t.source for t in traces))[:5]
            lines.append(f"  {typ}: {', '.join(sources)}")

        # Current goals
        todo_state = self.todo_manager.format_list(include_completed=False)
        if todo_state and "empty" not in todo_state.lower():
            lines.append("")
            lines.append(todo_state)

        return "\n".join(lines)

    # Keep the old name working as an alias
    build_recovery_notice = build_context_recovery_notice

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "compaction_traces": [
                {
                    "original_type": t.original_type,
                    "source": t.source,
                    "summary": t.summary,
                    "tokens_freed": t.tokens_freed,
                    "compacted_at": t.compacted_at,
                }
                for t in self.compaction_traces
            ],
            "node_metadata": [
                {
                    "node_id": m.node_id,
                    "first_seen": m.first_seen,
                    "last_seen": m.last_seen,
                    "seen_count": m.seen_count,
                    "role": m.role,
                    "msg_type": m.msg_type,
                    "source": m.source,
                    "last_tokens": m.last_tokens,
                    "lifecycle_state": m.lifecycle_state,
                }
                for m in self.node_metadata.values()
            ],
        }

    def load_dict(self, data: dict):
        raw = data.get("compaction_traces", data.get("eviction_traces", []))
        self.compaction_traces = []
        for t in raw:
            # Handle both old format (token_count) and new (tokens_freed)
            tokens = t.get("tokens_freed", t.get("token_count", 0))
            self.compaction_traces.append(
                CompactionTrace(
                    original_type=t["original_type"],
                    source=t["source"],
                    summary=t["summary"],
                    tokens_freed=tokens,
                    compacted_at=t.get("compacted_at", t.get("evicted_at", 0)),
                )
            )
        self.node_metadata = {}
        for m in data.get("node_metadata", []):
            try:
                node = ContextNodeMeta(
                    node_id=str(m.get("node_id", "")),
                    first_seen=float(m.get("first_seen", 0)),
                    last_seen=float(m.get("last_seen", 0)),
                    seen_count=int(m.get("seen_count", 0)),
                    role=str(m.get("role", "")),
                    msg_type=str(m.get("msg_type", "")),
                    source=str(m.get("source", "")),
                    last_tokens=int(m.get("last_tokens", 0)),
                    lifecycle_state=str(m.get("lifecycle_state", "warm")),
                )
                if node.node_id:
                    self.node_metadata[node.node_id] = node
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Phase 1: Duplicate consolidation
    # ------------------------------------------------------------------

    def _consolidate_duplicates(self, messages: List[Any]) -> Tuple[List[Any], int]:
        """Keep only the latest read of each file, replace older reads."""
        file_reads: Dict[str, List[int]] = {}  # normalised path -> [indices]

        for i, msg in enumerate(messages):
            if i in PROTECTED_INDICES:
                continue
            content = _get_content(msg)
            if not content or COMPACT_MARKER in content[:20]:
                continue

            match = re.match(r"\[read_file result\]\s*\n?(.+?)(?:\n|$)", content)
            if match:
                path = _normalize_path(match.group(1).strip())
                file_reads.setdefault(path, []).append(i)

        tokens_freed = 0
        for path, indices in file_reads.items():
            if len(indices) < 2:
                continue
            latest = max(indices)
            for idx in indices:
                if idx == latest:
                    continue
                msg = messages[idx]
                old_content = _get_content(msg)
                old_tokens = estimate_tokens(old_content)

                trace = CompactionTrace(
                    "duplicate_read",
                    path,
                    f"Superseded read of {path}",
                    tokens_freed=old_tokens,
                )
                notice = trace.format_notice()
                _set_content(msg, notice)
                freed = old_tokens - estimate_tokens(notice)
                if freed > 0:
                    tokens_freed += freed
                    self.compaction_traces.append(trace)

        return messages, tokens_freed

    # ------------------------------------------------------------------
    # Phase 2: Scoring
    # ------------------------------------------------------------------

    def _build_query_text(self) -> str:
        """Build a single query string from active todo context for embedding.

        Combines todo titles, descriptions, notes, and context_refs into one
        string that represents "what the agent is currently working on".
        """
        parts: List[str] = []
        for todo in self.todo_manager.list_active():
            if todo.title:
                parts.append(todo.title)
            if todo.description:
                parts.append(todo.description)
            if todo.notes:
                parts.append(todo.notes)
            for ref in todo.context_refs:
                parts.append(ref)
        return " | ".join(parts) if parts else ""

    def _get_protected_recent(self, messages: List[Any]) -> frozenset:
        """Compute the set of recent-window indices that are token-budget protected.

        Walks backward from the tail, accumulating tokens.  Once the
        cumulative total exceeds ``recent_window_max_tokens``, remaining
        older messages in the window are NOT protected — they become
        compaction/eviction candidates like any other message.

        Always protects at least the last 2 messages (current exchange).
        """
        total = len(messages)
        if total <= max(PROTECTED_INDICES) + 1:
            return frozenset()

        window_start = max(max(PROTECTED_INDICES) + 1, total - self.recent_window)
        protected: List[int] = []
        budget_remaining = self.recent_window_max_tokens
        min_protect = 2  # always protect at least the last exchange

        for i in range(total - 1, window_start - 1, -1):
            content = _get_content(messages[i])
            tok = estimate_tokens(content)
            if len(protected) < min_protect or budget_remaining >= tok:
                protected.append(i)
                budget_remaining -= tok
            # else: this message is too large to fit in the token budget,
            # leave it unprotected so it can be compacted/evicted

        return frozenset(protected)

    def _score_all_messages(
        self,
        messages: List[Any],
        ref_paths: Set[str],
        ref_keywords: Set[str],
    ) -> List[Tuple[int, float, str, str]]:
        """Score every non-protected message.

        Uses batch embedding for semantic relevance when the model is
        available, falling back to keyword matching otherwise.

        Returns ``[(index, keep_priority, msg_type, source), ...]``
        where ``keep_priority`` is 0.0 (evict first) to 1.0 (keep).
        """
        total = len(messages)
        protected_recent = self._get_protected_recent(messages)

        # First pass: collect candidates
        candidates: List[Tuple[int, str, int, str, str, str]] = []
        # (index, content, tokens, msg_type, source, role)

        for i, msg in enumerate(messages):
            if i in PROTECTED_INDICES or i in protected_recent:
                continue

            content = _get_content(msg)
            if not content or len(content) < 10:
                continue
            if COMPACT_MARKER in content[:20]:
                continue

            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            msg_type, source = self._classify(content, role)

            tokens = estimate_tokens(content)
            # Guidance nudges are always candidates (they accumulate as noise).
            # Other messages need a minimum size to be worth compacting.
            if tokens < MIN_COMPACT_TOKENS and msg_type != "guidance_nudge":
                continue

            candidates.append((i, content, tokens, msg_type, source, role))

        if not candidates:
            return []

        # Compute semantic relevance via batch embedding
        query_text = self._build_query_text()
        contents = [c[1] for c in candidates]

        if query_text:
            relevance_scores = self._scorer.score_batch(contents, query_text)
        else:
            # No active todos → neutral relevance for all
            relevance_scores = [0.4] * len(candidates)

        # Combine all signals into final keep-priority score
        results: List[Tuple[int, float, str, str]] = []
        base_scores: List[float] = []
        for (index, _content, tokens, msg_type, source, _role), relevance in zip(
            candidates, relevance_scores
        ):
            score = self._compute_score(
                msg_type,
                index,
                total,
                tokens,
                relevance,
            )
            base_scores.append(score)
            results.append((index, score, msg_type, source))

        # Learned prior: train fast logistic model on weak labels from this
        # candidate set and blend expected keep-probability into the final rank.
        learned_keep = self._infer_policy_keep_scores(
            candidates=candidates,
            relevance_scores=relevance_scores,
            total=total,
            query_text=query_text,
        )
        if learned_keep:
            blended: List[Tuple[int, float, str, str]] = []
            for index, base, msg_type, source in results:
                ml_keep = learned_keep.get(index, base)
                score = max(0.0, min(1.0, base * 0.65 + ml_keep * 0.35))
                blended.append((index, score, msg_type, source))
            return blended

        return results

    def _compute_score(
        self,
        msg_type: str,
        index: int,
        total: int,
        tokens: int,
        relevance: float,
    ) -> float:
        """Compute keep-priority: 0 = compact first, 1 = keep.

        ``relevance`` is pre-computed (either via embedding or keyword fallback).
        """

        # --- 1. Recency (0 = oldest non-protected, 1 = newest non-recent) ---
        effective_start = max(PROTECTED_INDICES) + 1  # 3
        effective_end = max(effective_start + 1, total - self.recent_window)
        recency = (
            (index - effective_start) / (effective_end - effective_start)
            if effective_end > effective_start
            else 1.0
        )

        # --- 2. Regeneration cost (how easy to get this content back?) ---
        # Increased costs for tool results to prevent aggressive spilling of recent work
        regen_costs = {
            "file_read": 0.3,  # re-read the file
            "command_output": 0.5,  # may contain important output, harder to reproduce
            "search_result": 0.45,  # search results are valuable, harder to re-run
            "assistant_analysis": 0.6,  # expensive — reasoning is non-deterministic
            "assistant_tool_call": 0.5,  # contains action context
            "todo_result": 0.05,  # trivially regenerated
            "context_result": 0.05,  # trivially regenerated
            "introspect_result": 0.15,  # reasoning already absorbed into subsequent actions
            "guidance_nudge": 0.01,  # lifecycle guidance should decay quickly
            "other_tool_result": 0.4,
            "other": 0.4,
        }
        regen_cost = regen_costs.get(msg_type, 0.4)

        # --- 2.5: Fresh tool result protection ---
        # Don't compact very recent tool results (last 2 messages)
        # These are likely still being actively used by the model
        if msg_type in ("command_output", "search_result", "other_tool_result"):
            # If this is one of the last 2 non-protected messages, boost its score significantly
            messages_from_end = total - index
            if messages_from_end <= 2 and index > max(PROTECTED_INDICES):
                regen_cost = min(
                    1.0, regen_cost + 0.4
                )  # Boost to protect recent results

        # --- 3. Size pressure: larger messages are better targets ---
        size_pressure = min(0.15, tokens / 20_000)

        # --- Combine ---
        score = (recency * 0.30 + relevance * 0.35 + regen_cost * 0.35) - size_pressure

        return max(0.0, min(1.0, score))

    @staticmethod
    def _policy_weak_label(msg_type: str, tokens: int, role: str) -> int:
        """Weak label mapping used by the runtime logistic prior.

        0=KEEP_FULL, 1=SUMMARIZE, 2=ARCHIVE_WITH_BREADCRUMB, 3=EVICT
        """
        if msg_type == "guidance_nudge":
            return 3
        if msg_type in ("todo_result", "context_result"):
            return 3
        if msg_type == "introspect_result":
            return 2 if tokens >= 120 else 1
        if msg_type in ("command_output", "search_result", "other_tool_result"):
            return 2 if tokens >= 120 else 1
        if msg_type == "assistant_analysis":
            return 1 if tokens >= 160 else 0
        if role == "system":
            return 0
        return 0

    def _infer_policy_keep_scores(
        self,
        candidates: List[Tuple[int, str, int, str, str, str]],
        relevance_scores: List[float],
        total: int,
        query_text: str,
    ) -> Dict[int, float]:
        """Return learned keep-scores keyed by message index.

        DISABLED: sklearn import causes Windows WMI hang.
        Returns empty dict to use fallback keyword-based scoring.
        """
        self.last_policy_stats = {
            "used": False,
            "reason": "sklearn_disabled_due_to_wmi_hang",
        }
        return {}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(content: str, role: str) -> Tuple[str, str]:
        """Classify a message into ``(type, source_label)``."""

        # Structured guidance markers injected by the agent.
        if content.startswith("[GUIDANCE_NUDGE"):
            m = re.search(r"type=([a-zA-Z0-9._-]+)", content)
            nudge_type = m.group(1) if m else "guidance"
            return "guidance_nudge", nudge_type

        # Backward compatibility for sessions without explicit marker.
        if role == "user":
            lower = content.lower()
            if "without using introspect" in lower:
                return "guidance_nudge", "introspect"
            if "consider using the introspect tool" in lower:
                return "guidance_nudge", "introspect-tip"
            if "result contains errors" in lower:
                return "guidance_nudge", "error-analysis"

        # Tool results are stored as user messages: "[tool_name result]\n..."
        tool_match = re.match(r"\[(\w+) result(?:\s*-[^\]]+)?\]", content)
        if tool_match:
            tool_name = tool_match.group(1)

            if tool_name == "read_file":
                lines = content.split("\n", 2)
                path = lines[1].strip() if len(lines) > 1 else "unknown"
                return "file_read", path

            if tool_name == "execute_command":
                cmd_match = re.search(r"\$\s*(.+?)(?:\n|$)", content)
                cmd = cmd_match.group(1).strip()[:80] if cmd_match else "command"
                return "command_output", cmd

            if tool_name in ("search_files", "grep_search"):
                return "search_result", "search"

            if tool_name == "manage_todos":
                return "todo_result", "todos"

            if tool_name in ("list_context", "remove_from_context"):
                return "context_result", "context"

            if tool_name == "introspect":
                return "introspect_result", "introspect"

            return "other_tool_result", tool_name

        # Introspect results (single-tool format — no [tool_name result] wrapper)
        if content.startswith("[Deep analysis complete"):
            return "introspect_result", "introspect"

        # Assistant messages
        if role == "assistant":
            has_tool_xml = bool(
                re.search(
                    r"<(?:read_file|write_to_file|replace_in_file|execute_command"
                    r"|search_files|list_files|manage_todos"
                    r"|list_context|remove_from_context|analyze_image|web_search"
                    r"|check_background_process|stop_background_process"
                    r"|list_background_processes|introspect)\b",
                    content,
                )
            )
            if has_tool_xml:
                return "assistant_tool_call", "tool_call"
            return "assistant_analysis", "analysis"

        return "other", ""

    # ------------------------------------------------------------------
    # Compaction (content replacement)
    # ------------------------------------------------------------------

    def _compact_message(
        self,
        content: str,
        msg_type: str,
        source: str,
        message_index: int = -1,
    ) -> Tuple[str, str]:
        """Create a compact replacement.

        Args:
            content: The original message content
            msg_type: Type of message (command_output, search_result, etc.)
            source: Source identifier (file path, command, etc.)
            message_index: Index of the message in the conversation (for result storage)

        Returns ``(notice_text, one_line_summary)``.
        """
        lines = content.splitlines()
        line_count = len(lines)

        if msg_type == "file_read":
            defs = _extract_definitions(lines)
            summary = f"{', '.join(defs[:5])}" if defs else f"{line_count} lines"
            trace = CompactionTrace("file_read", source, summary)
            return trace.format_notice(), summary

        if msg_type == "command_output":
            sig = [l.strip() for l in lines if l.strip() and not l.startswith("[")]
            if sig:
                summary = sig[0][:80]
                if len(sig) > 1:
                    summary += f" … {sig[-1][:40]}"
            else:
                summary = f"{line_count} lines of output"

            # Store full result before compacting (with error handling)
            result_id = None
            try:
                result_id = self.result_storage.store_result(
                    tool_name="execute_command",
                    content=content,
                    message_index=message_index,
                )
            except Exception as e:
                log.warning(f"Failed to store command output for retrieval: {e}")

            if result_id:
                notice = (
                    f"[{COMPACT_MARKER} Command output: {summary}]\n"
                    f"Full result stored as {result_id}. Use retrieve_tool_result to access."
                )
            else:
                notice = (
                    f"[{COMPACT_MARKER} Command output: {summary}]\n"
                    f"(Result storage failed - re-run command to see full output)"
                )
            return notice, summary

        if msg_type == "search_result":
            match_count = sum(1 for l in lines if ":" in l and re.search(r":\d+:", l))
            summary = f"{match_count} matches"

            # Store full result before compacting (with error handling)
            result_id = None
            try:
                result_id = self.result_storage.store_result(
                    tool_name="search", content=content, message_index=message_index
                )
            except Exception as e:
                log.warning(f"Failed to store search results for retrieval: {e}")

            if result_id:
                notice = (
                    f"[{COMPACT_MARKER} Search results: {summary}]\n"
                    f"Full result stored as {result_id}. Use retrieve_tool_result to access."
                )
            else:
                notice = (
                    f"[{COMPACT_MARKER} Search results: {summary}]\n"
                    f"(Result storage failed - re-run search to see full results)"
                )
            return notice, summary

        if msg_type == "assistant_tool_call":
            # Keep the XML tool call, compress the reasoning before it.
            xml_match = re.search(
                r"(<(?:read_file|write_to_file|replace_in_file|execute_command"
                r"|search_files|list_files|manage_todos"
                r"|list_context|remove_from_context|analyze_image|web_search"
                r"|check_background_process|stop_background_process"
                r"|list_background_processes)\b.*)",
                content,
                re.DOTALL,
            )
            if xml_match:
                xml_part = xml_match.group(1)
                pre_xml = content[: xml_match.start()].strip()
                if pre_xml:
                    first_sentence = re.split(r"[.\n]", pre_xml)[0][:120]
                    summary = first_sentence
                    notice = f"[{COMPACT_MARKER} {first_sentence}]\n\n{xml_part}"
                else:
                    summary = "tool call"
                    notice = xml_part
                return notice, summary
            # fallthrough
            summary = lines[0][:120] if lines else "assistant response"
            trace = CompactionTrace("assistant", source, summary)
            return trace.format_notice(), summary

        if msg_type == "assistant_analysis":
            if len(content) >= _LONG_ASSISTANT_COMPACT_CHARS:
                result_id = None
                try:
                    result_id = self.result_storage.store_result(
                        tool_name="assistant_response",
                        content=content,
                        message_index=message_index,
                    )
                except Exception as e:
                    log.warning(
                        f"Failed to store assistant response for retrieval: {e}"
                    )

                excerpt = _sample_start_middle_end(content)
                summary_line = next(
                    (ln.strip() for ln in lines if ln.strip()),
                    "long assistant response",
                )
                summary = summary_line[:120]

                if result_id:
                    notice = (
                        f"[{COMPACT_MARKER} Assistant response (abbreviated): {summary}]\n"
                        f"Full response stored as {result_id}. Use retrieve_tool_result to access.\n\n"
                        f"{excerpt}"
                    )
                else:
                    notice = (
                        f"[{COMPACT_MARKER} Assistant response (abbreviated): {summary}]\n"
                        f"(Result storage failed - full response not retrievable)\n\n"
                        f"{excerpt}"
                    )
                return notice, summary

            # Keep just the first meaningful sentence.
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith(
                    ("<", "#", "---", "```", "**", "//")
                ):
                    summary = stripped[:150]
                    break
            else:
                summary = "analysis"
            trace = CompactionTrace("assistant", "reasoning", summary)
            return trace.format_notice(), summary

        if msg_type in ("todo_result", "context_result"):
            summary = f"{msg_type} output"
            trace = CompactionTrace(msg_type, source, summary)
            return trace.format_notice(), summary

        if msg_type == "introspect_result":
            focus_match = re.search(r"FOCUS:\s*(.+?)(?:\n|$)", content)
            focus_hint = focus_match.group(1).strip()[:80] if focus_match else ""
            first_lines = content.split("\n", 6)
            excerpt = " ".join(l.strip() for l in first_lines[2:6] if l.strip())[:200]
            summary = focus_hint or excerpt or "deep analysis"
            return (
                f"[{COMPACT_MARKER} Introspect: {summary}]\n"
                f"(Reasoning already applied in subsequent actions)"
            ), summary

        if msg_type == "guidance_nudge":
            summary = f"guidance:{source}"
            return f"[{COMPACT_MARKER} Guidance note archived: {source}]", summary

        if msg_type == "other_tool_result":
            # Extract tool name from content (format: [tool_name result])
            tool_match = re.match(r"\[(\w+) result", content)
            tool_name = tool_match.group(1) if tool_match else source

            # Store full result before compacting (with error handling)
            result_id = None
            try:
                result_id = self.result_storage.store_result(
                    tool_name=tool_name, content=content, message_index=message_index
                )
            except Exception as e:
                log.warning(f"Failed to store tool result for {tool_name}: {e}")

            summary = lines[0][:80] if lines else f"{tool_name} result"
            if result_id:
                notice = (
                    f"[{COMPACT_MARKER} {tool_name}: {summary}]\n"
                    f"Full result stored as {result_id}. Use retrieve_tool_result to access."
                )
            else:
                notice = (
                    f"[{COMPACT_MARKER} {tool_name}: {summary}]\n"
                    f"(Result storage failed - re-run to see full output)"
                )
            return notice, summary

        # Generic fallback
        summary = lines[0][:100] if lines else "content"
        trace = CompactionTrace(msg_type, source, summary)
        return trace.format_notice(), summary

    def _compact_stale_guidance(
        self,
        messages: List[Any],
        limit: Optional[int] = None,
    ) -> Tuple[List[Any], int, int]:
        """Compact stale guidance/nudge messages outside the recent window."""
        tokens_freed = 0
        compacted = 0
        protected_recent = self._get_protected_recent(messages)
        max_to_prune = limit if limit is not None else self.max_guidance_prune_per_tick

        for i, msg in enumerate(messages):
            if compacted >= max_to_prune:
                break
            if i in PROTECTED_INDICES or i in protected_recent:
                continue
            content = _get_content(msg)
            if not content or COMPACT_MARKER in content[:20]:
                continue
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            msg_type, source = self._classify(content, role)
            if msg_type != "guidance_nudge":
                continue

            old_tokens = estimate_tokens(content)
            notice, summary = self._compact_message(content, msg_type, source, i)
            new_tokens = estimate_tokens(notice)
            freed = old_tokens - new_tokens
            if freed <= 0:
                continue

            _set_content(msg, notice)
            tokens_freed += freed
            compacted += 1
            self.compaction_traces.append(
                CompactionTrace(
                    original_type="guidance_nudge",
                    source=source,
                    summary=summary,
                    tokens_freed=freed,
                )
            )

        if compacted and len(self.compaction_traces) > self._max_traces:
            self.compaction_traces = self.compaction_traces[-self._max_traces :]
        return messages, tokens_freed, compacted


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_content(msg: Any) -> str:
    """Extract string content from a message (object or dict)."""
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
    return content if isinstance(content, str) else ""


def _set_content(msg: Any, content: str) -> None:
    """Set string content on a message (object or dict)."""
    if hasattr(msg, "content"):
        msg.content = content
    else:
        msg["content"] = content


def _extract_definitions(lines: List[str], limit: int = 150) -> List[str]:
    """Pull out function/class names from source code lines."""
    defs: List[str] = []
    for line in lines[:limit]:
        s = line.strip()
        if s.startswith(
            ("def ", "class ", "func ", "function ", "export ", "type ", "interface ")
        ):
            name = s.split("(")[0].split("{")[0].split(":")[0].strip()
            defs.append(name)
    return defs


def _normalize_path(path: str) -> str:
    """Normalize a file path for dedup comparison.

    Handles Windows case-insensitivity, forward/back-slash mixing,
    and strips leading drive letters for comparison.
    """
    import os

    p = path.replace("\\", "/").strip()
    # Case-insensitive on Windows
    if os.name == "nt":
        p = p.lower()
    # Strip trailing slashes
    p = p.rstrip("/")
    return p


def _sample_start_middle_end(content: str, chunk_chars: int = 450) -> str:
    """Build an abbreviated view using start/middle/end excerpts."""
    text = content.strip()
    if len(text) <= chunk_chars * 3 + 40:
        return text

    n = len(text)
    start = text[:chunk_chars].rstrip()
    mid_start = max(0, (n // 2) - (chunk_chars // 2))
    middle = text[mid_start : mid_start + chunk_chars].strip()
    end = text[-chunk_chars:].lstrip()

    return (
        "[Start excerpt]\n"
        f"{start}\n\n"
        "[Middle excerpt]\n"
        f"{middle}\n\n"
        "[End excerpt]\n"
        f"{end}"
    )


def _message_fingerprint(role: str, content: str) -> str:
    """Stable fingerprint for message-level metadata tracking."""
    base = f"{role}\n{content[:4000]}"
    return hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()
