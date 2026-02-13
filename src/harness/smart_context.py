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
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Set, TYPE_CHECKING

import numpy as np

from .context_management import estimate_tokens, estimate_messages_tokens

if TYPE_CHECKING:
    from .todo_manager import TodoManager


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
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
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
            vecs = self._model.encode(batch_texts, normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=False)
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
        self, contents: List[str], query: str,
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
        query_words = set(re.findall(r'\b[a-zA-Z_]\w{2,}\b', query.lower()))
        noise = {
            "the", "and", "for", "this", "that", "with", "from", "are", "was",
            "has", "have", "not", "but", "can", "all", "will", "use", "add",
            "into", "also", "any", "each", "get", "set", "should", "would",
            "need", "make", "like", "new", "see", "now", "just", "its", "our",
            "file", "code", "test", "run", "check", "fix", "error", "result",
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
    original_type: str       # 'file_read', 'command_output', 'search_result', 'assistant', ...
    source: str              # file path, command string, or brief label
    summary: str             # one-line description of what was there
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

        if current_tokens <= budget:
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

        tokens_still_needed = excess - total_freed
        compacted_count = 0

        for index, _score, msg_type, source in scored:
            if tokens_still_needed <= 0:
                break

            msg = messages[index]
            content = _get_content(msg)
            old_tokens = estimate_tokens(content)

            notice, summary = self._compact_message(content, msg_type, source)
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
            self.compaction_traces = self.compaction_traces[-self._max_traces:]

        if compacted_count > 0:
            compact_freed = total_freed - dup_freed
            report_parts.append(
                f"Compacted {compacted_count} messages: -{compact_freed:,} tok"
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
                    self.compaction_traces.append(CompactionTrace(
                        original_type=msg_type,
                        source=source,
                        summary=summary,
                        tokens_freed=freed,
                    ))

            if evicted_indices:
                messages = [
                    m for i, m in enumerate(messages)
                    if i not in evicted_indices
                ]
                report_parts.append(
                    f"Evicted {len(evicted_indices)} low-priority messages"
                )

        # Trim trace history
        if len(self.compaction_traces) > self._max_traces:
            self.compaction_traces = self.compaction_traces[-self._max_traces:]

        return messages, total_freed, "; ".join(report_parts)

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
        }

    def load_dict(self, data: dict):
        raw = data.get("compaction_traces", data.get("eviction_traces", []))
        self.compaction_traces = []
        for t in raw:
            # Handle both old format (token_count) and new (tokens_freed)
            tokens = t.get("tokens_freed", t.get("token_count", 0))
            self.compaction_traces.append(CompactionTrace(
                original_type=t["original_type"],
                source=t["source"],
                summary=t["summary"],
                tokens_freed=tokens,
                compacted_at=t.get("compacted_at", t.get("evicted_at", 0)),
            ))

    # ------------------------------------------------------------------
    # Phase 1: Duplicate consolidation
    # ------------------------------------------------------------------

    def _consolidate_duplicates(
        self, messages: List[Any]
    ) -> Tuple[List[Any], int]:
        """Keep only the latest read of each file, replace older reads."""
        file_reads: Dict[str, List[int]] = {}  # normalised path -> [indices]

        for i, msg in enumerate(messages):
            if i in PROTECTED_INDICES:
                continue
            content = _get_content(msg)
            if not content or COMPACT_MARKER in content[:20]:
                continue

            match = re.match(r'\[read_file result\]\s*\n?(.+?)(?:\n|$)', content)
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
                    "duplicate_read", path,
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
        candidates: List[Tuple[int, str, int, str, str]] = []
        # (index, content, tokens, msg_type, source)

        for i, msg in enumerate(messages):
            if i in PROTECTED_INDICES or i in protected_recent:
                continue

            content = _get_content(msg)
            if not content or len(content) < 50:
                continue
            if COMPACT_MARKER in content[:20]:
                continue

            tokens = estimate_tokens(content)
            if tokens < MIN_COMPACT_TOKENS:
                continue

            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            msg_type, source = self._classify(content, role)
            candidates.append((i, content, tokens, msg_type, source))

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
        for (index, content, tokens, msg_type, source), relevance in zip(
            candidates, relevance_scores
        ):
            score = self._compute_score(
                msg_type, index, total, tokens, relevance,
            )
            results.append((index, score, msg_type, source))

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
        effective_start = max(PROTECTED_INDICES) + 1          # 3
        effective_end = max(effective_start + 1, total - self.recent_window)
        recency = (
            (index - effective_start) / (effective_end - effective_start)
            if effective_end > effective_start
            else 1.0
        )

        # --- 2. Regeneration cost (how easy to get this content back?) ---
        regen_costs = {
            "file_read":          0.2,   # just re-read the file
            "command_output":     0.10,  # cheap — re-run or read spill file
            "search_result":      0.15,  # re-search is easy
            "assistant_analysis": 0.6,   # expensive — reasoning is non-deterministic
            "assistant_tool_call":0.5,   # contains action context
            "todo_result":        0.05,  # trivially regenerated
            "context_result":     0.05,  # trivially regenerated
            "other_tool_result":  0.25,
            "other":              0.4,
        }
        regen_cost = regen_costs.get(msg_type, 0.4)

        # --- 3. Size pressure: larger messages are better targets ---
        size_pressure = min(0.15, tokens / 20_000)

        # --- Combine ---
        score = (
            recency * 0.30
            + relevance * 0.35
            + regen_cost * 0.35
        ) - size_pressure

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(content: str, role: str) -> Tuple[str, str]:
        """Classify a message into ``(type, source_label)``."""

        # Tool results are stored as user messages: "[tool_name result]\n..."
        tool_match = re.match(r'\[(\w+) result(?:\s*-[^\]]+)?\]', content)
        if tool_match:
            tool_name = tool_match.group(1)

            if tool_name == "read_file":
                lines = content.split("\n", 2)
                path = lines[1].strip() if len(lines) > 1 else "unknown"
                return "file_read", path

            if tool_name == "execute_command":
                cmd_match = re.search(r'\$\s*(.+?)(?:\n|$)', content)
                cmd = cmd_match.group(1).strip()[:80] if cmd_match else "command"
                return "command_output", cmd

            if tool_name in ("search_files", "grep_search"):
                return "search_result", "search"

            if tool_name == "manage_todos":
                return "todo_result", "todos"

            if tool_name in ("list_context", "remove_from_context"):
                return "context_result", "context"

            return "other_tool_result", tool_name

        # Assistant messages
        if role == "assistant":
            has_tool_xml = bool(re.search(
                r'<(?:read_file|write_to_file|replace_in_file|execute_command'
                r'|search_files|list_files|manage_todos|attempt_completion'
                r'|list_context|remove_from_context|analyze_image|web_search'
                r'|check_background_process|stop_background_process'
                r'|list_background_processes|introspect)\b',
                content,
            ))
            if has_tool_xml:
                return "assistant_tool_call", "tool_call"
            return "assistant_analysis", "analysis"

        return "other", ""

    # ------------------------------------------------------------------
    # Compaction (content replacement)
    # ------------------------------------------------------------------

    def _compact_message(
        self, content: str, msg_type: str, source: str,
    ) -> Tuple[str, str]:
        """Create a compact replacement.

        Returns ``(notice_text, one_line_summary)``.
        """
        lines = content.splitlines()
        line_count = len(lines)

        if msg_type == "file_read":
            defs = _extract_definitions(lines)
            summary = (
                f"{', '.join(defs[:5])}" if defs else f"{line_count} lines"
            )
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
            trace = CompactionTrace("command_output", source, summary)
            return trace.format_notice(), summary

        if msg_type == "search_result":
            match_count = sum(
                1 for l in lines if ":" in l and re.search(r":\d+:", l)
            )
            summary = f"{match_count} matches"
            trace = CompactionTrace("search_result", source, summary)
            return trace.format_notice(), summary

        if msg_type == "assistant_tool_call":
            # Keep the XML tool call, compress the reasoning before it.
            xml_match = re.search(
                r'(<(?:read_file|write_to_file|replace_in_file|execute_command'
                r'|search_files|list_files|manage_todos|attempt_completion'
                r'|list_context|remove_from_context|analyze_image|web_search'
                r'|check_background_process|stop_background_process'
                r'|list_background_processes)\b.*)',
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

        # Generic fallback
        summary = lines[0][:100] if lines else "content"
        trace = CompactionTrace(msg_type, source, summary)
        return trace.format_notice(), summary


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
        if s.startswith(("def ", "class ", "func ", "function ", "export ", "type ", "interface ")):
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
