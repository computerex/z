"""Quality benchmark for live SmartContextManager pruning behavior.

This evaluates whether context management removes low-value history while
retaining high-value context, using two proxies:
1) weak policy labels (KEEP_FULL/SUMMARIZE/ARCHIVE/EVICT)
2) near-future lexical reference overlap
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context_management import estimate_messages_tokens
from .context_replay import _weak_label
from .smart_context import COMPACT_MARKER, SmartContextManager, _message_fingerprint
from .todo_manager import TodoManager


PROTECTED_INDICES = {0, 1, 2}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will",
    "your", "about", "into", "after", "before", "there", "their", "would",
    "should", "could", "using", "without", "where", "when", "what", "which",
    "while", "then", "than", "been", "being", "were", "was", "are", "is",
    "to", "of", "in", "on", "at", "as", "an", "a", "it", "or", "if", "by",
    "be", "we", "you", "they", "he", "she", "i", "my", "our", "us",
}


def _text_from_message(msg: Dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _token_set(text: str) -> set[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{3,}", text)]
    return {w for w in words if w not in STOPWORDS}


def _needed_later_proxy(messages: Sequence[Dict[str, Any]], lookahead: int = 30) -> List[bool]:
    texts = [_text_from_message(m) for m in messages]
    tokens = [_token_set(t) for t in texts]
    needed = [False] * len(messages)
    for i, t in enumerate(tokens):
        if not t:
            continue
        best_shared = 0
        best_jaccard = 0.0
        for j in range(i + 1, min(len(messages), i + 1 + lookahead)):
            u = tokens[j]
            if not u:
                continue
            shared = len(t & u)
            if shared > best_shared:
                best_shared = shared
            union = len(t | u)
            if union:
                jac = shared / union
                if jac > best_jaccard:
                    best_jaccard = jac
        needed[i] = best_shared >= 4 or (best_shared >= 3 and best_jaccard >= 0.08)
    return needed


def _run_single_budget(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    overrides: Optional[Dict[str, Any]] = None,
    future_lookahead: int = 30,
) -> Dict[str, Any]:
    tm = TodoManager()
    scm = SmartContextManager(tm)
    if overrides:
        for k, v in overrides.items():
            if hasattr(scm, k):
                setattr(scm, k, v)
    before = estimate_messages_tokens(messages)
    out, freed, report = scm.semantic_maintenance_tick(
        copy.deepcopy(messages),
        max_tokens=max_tokens,
        current_tokens=before,
    )
    after = estimate_messages_tokens(out)

    needed = _needed_later_proxy(messages, lookahead=future_lookahead)

    after_fp = {
        _message_fingerprint(m.get("role", "user"), _text_from_message(m))
        for m in out
    }
    outcome_counts = Counter()
    by_type = Counter()
    by_weak = Counter()

    pruned_needed = 0
    pruned_unneeded = 0
    kept_needed = 0
    kept_unneeded = 0

    for i, m in enumerate(messages):
        if i in PROTECTED_INDICES:
            continue
        role = m.get("role", "user")
        content = _text_from_message(m)
        fp = _message_fingerprint(role, content)
        msg_type, _source = scm._classify(content, role)
        weak = _weak_label(msg_type, len(content) // 4, role)

        same_idx_compacted = False
        if i < len(out):
            after_content = _text_from_message(out[i])
            same_idx_compacted = (
                isinstance(after_content, str)
                and after_content.startswith(f"[{COMPACT_MARKER}")
                and after_content != content
            )

        if same_idx_compacted:
            outcome = "compacted"
        elif fp in after_fp:
            outcome = "kept"
        else:
            outcome = "evicted"

        outcome_counts[outcome] += 1
        by_type[(outcome, msg_type)] += 1
        by_weak[(outcome, weak)] += 1

        if outcome in ("compacted", "evicted"):
            if needed[i]:
                pruned_needed += 1
            else:
                pruned_unneeded += 1
        else:
            if needed[i]:
                kept_needed += 1
            else:
                kept_unneeded += 1

    pruned = outcome_counts["compacted"] + outcome_counts["evicted"]
    total_needed = sum(
        1 for i in range(len(messages)) if i not in PROTECTED_INDICES and needed[i]
    )
    total_unneeded = sum(
        1 for i in range(len(messages)) if i not in PROTECTED_INDICES and not needed[i]
    )

    keepfull_total = sum(v for (o, lbl), v in by_weak.items() if lbl == "KEEP_FULL")
    keepfull_kept = sum(
        v for (o, lbl), v in by_weak.items() if o == "kept" and lbl == "KEEP_FULL"
    )
    evict_total = sum(v for (o, lbl), v in by_weak.items() if lbl == "EVICT")
    evict_pruned = sum(
        v for (o, lbl), v in by_weak.items() if o in ("compacted", "evicted") and lbl == "EVICT"
    )

    return {
        "max_tokens": max_tokens,
        "before_tokens": before,
        "after_tokens": after,
        "tokens_freed_reported": freed,
        "token_reduction_pct": round((before - after) * 100.0 / max(1, before), 2),
        "report": report,
        "counts": {
            "kept": outcome_counts["kept"],
            "compacted": outcome_counts["compacted"],
            "evicted": outcome_counts["evicted"],
            "pruned": pruned,
        },
        "quality_weak_labels": {
            "keepfull_retention": round(keepfull_kept / max(1, keepfull_total), 4),
            "evict_capture": round(evict_pruned / max(1, evict_total), 4),
            "keepfull_kept": keepfull_kept,
            "keepfull_total": keepfull_total,
            "evict_pruned": evict_pruned,
            "evict_total": evict_total,
        },
        "quality_future_ref_proxy": {
            "needed_retention": round(kept_needed / max(1, total_needed), 4),
            "prune_precision_unneeded": round(pruned_unneeded / max(1, pruned), 4) if pruned else None,
            "unneeded_prune_recall": round(pruned_unneeded / max(1, total_unneeded), 4),
            "pruned_needed": pruned_needed,
            "pruned_unneeded": pruned_unneeded,
            "needed_total": total_needed,
            "unneeded_total": total_unneeded,
        },
        "top_pruned_types": [
            {"outcome": o, "msg_type": t, "count": c}
            for (o, t), c in sorted(
                ((k, v) for k, v in by_type.items() if k[0] in ("compacted", "evicted")),
                key=lambda x: x[1],
                reverse=True,
            )[:12]
        ],
        "policy_stats": scm.last_policy_stats,
    }


def run_benchmark(
    dump_path: Path,
    budgets: Sequence[int],
    overrides: Optional[Dict[str, Any]] = None,
    future_lookahead: int = 30,
) -> Dict[str, Any]:
    data = json.loads(dump_path.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Invalid dump format: 'messages' should be a list.")

    results = [
        _run_single_budget(
            messages,
            b,
            overrides=overrides,
            future_lookahead=future_lookahead,
        )
        for b in budgets
    ]
    return {
        "dump": str(dump_path),
        "message_count": len(messages),
        "budgets": list(budgets),
        "overrides": overrides or {},
        "future_lookahead": future_lookahead,
        "results": results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SmartContext quality on a context dump.")
    p.add_argument("--dump", required=True, help="Path to context dump JSON")
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[98000, 70000, 50000, 40000, 30000],
        help="Max-token budgets to evaluate",
    )
    p.add_argument(
        "--future-lookahead",
        type=int,
        default=30,
        help="Lookahead window for needed-later proxy metric",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override SmartContextManager attr as key=value (repeatable)",
    )
    p.add_argument("--out", default="", help="Optional JSON output path")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    overrides: Dict[str, Any] = {}
    for item in args.override:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        key = k.strip()
        raw = v.strip()
        if not key:
            continue
        # Basic scalar coercion.
        if raw.lower() in ("true", "false"):
            overrides[key] = raw.lower() == "true"
        else:
            try:
                if "." in raw:
                    overrides[key] = float(raw)
                else:
                    overrides[key] = int(raw)
            except ValueError:
                overrides[key] = raw

    result = run_benchmark(
        Path(args.dump),
        args.budgets,
        overrides=overrides,
        future_lookahead=args.future_lookahead,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote benchmark: {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
