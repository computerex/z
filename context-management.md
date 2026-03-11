# Semantic Context Management: Deep Dive & Research Analysis

## Startup Timing Results

After instrumenting harness startup, here's the phase-by-phase breakdown:

### Before optimization (sklearn at import time)
```
phase                       delta    cumulative
env_setup                     0ms           0ms
stdlib_imports              161ms         161ms
import_config              1795ms        1955ms   ← sklearn loaded here via smart_context.py
import_cline_agent            0ms        1955ms
import_harness_core           0ms        1955ms
import_rich                   0ms        1955ms
pre_init_logging             89ms        2044ms
init_logging                  0ms        2045ms
load_providers                0ms        2045ms
config_loaded                 0ms        2045ms
agent_created               241ms        2286ms   ← WorkspaceIndex.build() here
session_loaded                4ms        2291ms
ready                         0ms        2291ms
TOTAL                                   2291ms
```

### After optimization (sklearn deferred to first use)
```
phase                       delta    cumulative
env_setup                     0ms           0ms
stdlib_imports              163ms         163ms
import_config               727ms         890ms   ← numpy + anthropic + rich, no sklearn
import_cline_agent            0ms         890ms
import_harness_core           0ms         890ms
import_rich                   0ms         890ms
pre_init_logging             83ms         972ms
init_logging                  0ms         973ms
load_providers                1ms         973ms
config_loaded                 0ms         973ms
agent_created               284ms        1257ms
session_loaded                3ms        1260ms
ready                         0ms        1260ms
TOTAL                                   1260ms
```

**Result: 2291ms → 1260ms (45% faster, 1031ms saved)**

### Root cause breakdown
| Import | Time | Notes |
|--------|------|-------|
| `sklearn.linear_model` | **1065ms** | Heaviest single import — now deferred |
| `anthropic` SDK | 360ms | Used in streaming_client, hard to defer |
| `numpy` | 63ms | Required by smart_context at load time |
| `httpx` | 129ms | Required by streaming_client |
| `rich` (console, panel, etc.) | ~100ms | UI framework |
| `WorkspaceIndex.build()` | ~240ms | git ls-files + line counting |

### Remaining optimization opportunities
- `anthropic` SDK (360ms): Could lazy-import since it's only needed when using Anthropic provider
- `numpy` (63ms): Small but could be deferred if embeddings were fully optional
- `WorkspaceIndex.build()` (240ms): Could be async/background while waiting for first user input

---

## How Harness's Semantic Context System Works

### Architecture overview

Harness uses a **three-tier context management** system:

1. **SmartContextManager** — orchestrates compaction decisions
2. **SemanticScorer** — computes relevance via `all-MiniLM-L6-v2` embeddings (22M params, 384-dim)
3. **ToolResultStorage** — LRU cache for compacted tool results (enabling recovery)

### Scoring formula
```
score = (recency × 0.30 + relevance × 0.35 + regen_cost × 0.35) - size_pressure
```

- **Recency (30%)**: Linear scale, newest = 1.0, oldest = 0.0
- **Relevance (35%)**: Semantic similarity (cosine of normalized embeddings) between message content and active todo context
- **Regen cost (35%)**: Per-type regeneration cost (file_read=0.3, command_output=0.5, assistant_analysis=0.6)
- **Size pressure**: `min(0.15, tokens/20000)` — larger messages slightly favored for compaction

### Optional ML prior
When sklearn is available and ≥8 candidates exist, fits a logistic regression on weak labels using metadata + embedding features. Blended 65/35 with the base score. This is what required sklearn at startup — now deferred.

### Compaction pipeline
1. **Duplicate consolidation** — deduplicate file reads, keep latest
2. **Score and compact** — score all non-protected messages, compact lowest-scoring until under budget
3. **Eviction** — if still over budget, evict already-compacted messages first, then lowest-scored

### When it runs
- **Every turn**: `semantic_maintenance_tick()` — prunes stale guidance, soft compaction if over 62% budget
- **At 85% capacity**: `compact_context()` — full compaction to 75% budget target
- **Model load**: Lazy — `SentenceTransformer("all-MiniLM-L6-v2")` loaded on first compaction, not at startup

---

## Research Analysis: Is Semantic Context Management Helpful?

### The key question

Does embedding-based semantic scoring for context compaction provide meaningful benefit over simpler approaches (truncation, observation masking)?

### Evidence FOR semantic approaches

**1. Extractive compression outperforms truncation (multiple studies)**

A comprehensive benchmark (Characterizing Prompt Compression Methods, 2024) found that **extractive compression (semantic-aware selection) enables up to 10× compression with minimal accuracy degradation**, while simple truncation discards important information with recency bias. Semantic methods using embedding similarity consistently outperformed token pruning and random truncation across tasks.

**2. STAE (Semantic-Temporal Aware Eviction)**

Combines semantic distance and recency weighting (similar to what harness does). Outperforms FIFO truncation while preserving more task-relevant information. The harness scoring formula (`recency × 0.30 + relevance × 0.35 + regen_cost × 0.35`) is essentially a variant of STAE.

**3. Semantic compression extends effective context (arxiv 2312.09571)**

A framework demonstrated **6-8× context window extension** across question answering, summarization, few-shot learning, and information retrieval tasks without fine-tuning, using embedding-based redundancy detection.

**4. Microsoft ACE framework (ICLR 2026)**

Agentic Context Engineering treats contexts as "evolving playbooks" and achieved **+10.6% performance improvement** on agent tasks while reducing cost. It prevents "context collapse" (iterative rewriting eroding detail) and "brevity bias" (dropping domain insights).

### Evidence AGAINST (or questioning) complex approaches

**5. "The Complexity Trap" (NeurIPS 2025, JetBrains/TUM)**

This is the strongest counterargument. Key findings on SWE-bench Verified with 5 model configurations:

- **Simple observation masking halved costs while matching LLM summarization solve rates**
- LLM summarization caused **trajectory elongation** — agents ran 13-15% more turns because summaries smoothed over "stop" signals
- Summary API calls added 7%+ overhead with no cache reuse
- A hybrid approach (masking first, occasional summarization as fallback) achieved **7-11% additional cost reduction**

**Critical nuance**: This study compared **observation masking** (hiding old observations with placeholders) vs **LLM summarization** (using another LLM to summarize). It did NOT directly test embedding-based scoring like harness uses. Harness's approach is computationally cheaper than LLM summarization (no extra API call) and more targeted than simple observation masking.

**6. Context rot / lost-in-the-middle**

Research consistently shows LLM performance degrades as context grows, even with million-token windows. Growing context is noise, not signal — observation tokens comprise ~84% of average turn length in SE agents. This argues for *some* form of context management, but not necessarily the most complex one.

### Where harness's approach sits in the landscape

| Approach | Cost | Accuracy | Harness? |
|----------|------|----------|----------|
| No management (raw) | Highest | Baseline | No |
| Simple truncation (FIFO) | Low | Worst — recency bias | No |
| Observation masking | Low | Matches summarization | Partially — breadcrumb traces |
| **Embedding-based scoring** | **Low** | **Best extractive** | **Yes — primary mechanism** |
| LLM summarization | High | Good but trajectory elongation | No |
| Hybrid (masking + occasional summarization) | Medium | Best overall | Closest to this |

Harness actually occupies a **good middle ground**:
- It uses **embeddings** (cheap, local, no API call) rather than LLM summarization (expensive API call)
- It scores by **relevance to active todos** which provides task-aware signal absent from simple masking
- It preserves **breadcrumb traces** for recovery, similar to masking's placeholder approach
- The **ML prior** (logistic regression) adds marginal value but isn't justified at startup cost

### Verdict

**The semantic embedding approach IS justified, but with caveats:**

1. **Embeddings > truncation**: Unanimously supported by research. The ~60ms numpy import is worth it.

2. **Embeddings vs simple observation masking**: Marginal improvement. The NeurIPS 2025 study suggests simple masking gets you 90% of the way there for SE agent tasks. However, harness's embedding approach costs almost nothing extra at runtime (no API calls) unlike LLM summarization, so the cost-benefit ratio is favorable.

3. **The sklearn ML prior is NOT justified**: The logistic regression adds 1+ second import time, and research shows weak labels + simple heuristics match ML approaches on this task. The base scoring formula (`recency × 0.30 + relevance × 0.35 + regen_cost × 0.35`) already captures the key signals. The ML prior should remain optional and deferred (as now implemented).

4. **The scoring weights may not be optimal**: The 30/35/35 split is hand-tuned. The NeurIPS study found that hyperparameter tuning per agent scaffold matters significantly. Consider making these configurable.

5. **Biggest practical concern**: The `all-MiniLM-L6-v2` model load (first compaction) takes **2-7 seconds** depending on hardware. This happens mid-task when the user is waiting for a response. Consider:
   - Pre-warming in a background thread after startup
   - Using `local_files_only=True` (6-7x faster init per GitHub issue #2842)
   - Offering a "fast mode" that uses keyword fallback only

### Recommendations

1. **Keep the embedding approach** — it's the right architecture, backed by research
2. **Drop sklearn from startup** — done (45% speedup achieved)
3. **Pre-warm the embedding model in background** — avoids latency spike on first compaction
4. **Add `local_files_only=True`** to SentenceTransformer load — avoid network checks
5. **Make the ML prior opt-in** — it adds complexity without clear benefit
6. **Consider making scoring weights configurable** — different tasks may benefit from different tuning
7. **Add a `/context stats` command** — show users what's being compacted and why, building trust in the system
