"""Context-management benchmark pack with policy ablations.

Runs multiple SmartContext policy profiles across one or more dumps and
produces a ranked comparison using a composite quality score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .context_quality_benchmark import run_benchmark


DEFAULT_BUDGETS = [98000, 70000, 50000]

# Profiles to compare. "current" uses code defaults (no overrides).
PROFILES: Dict[str, Dict[str, Any]] = {
    "current": {},
    "conservative": {
        "soft_target_ratio": 0.62,
        "max_guidance_prune_per_tick": 8,
        "guidance_prune_boost_threshold": 0.55,
        "max_guidance_prune_per_tick_boosted": 12,
    },
    "balanced": {
        "soft_target_ratio": 0.55,
        "max_guidance_prune_per_tick": 12,
        "guidance_prune_boost_threshold": 0.50,
        "max_guidance_prune_per_tick_boosted": 16,
    },
    "aggressive": {
        "soft_target_ratio": 0.50,
        "max_guidance_prune_per_tick": 16,
        "guidance_prune_boost_threshold": 0.45,
        "max_guidance_prune_per_tick_boosted": 24,
    },
}


def _weighted_means(result: Dict[str, Any]) -> Dict[str, float]:
    rows = result.get("results", [])
    if not rows:
        return {
            "token_reduction": 0.0,
            "keepfull_retention": 0.0,
            "evict_capture": 0.0,
            "needed_retention": 0.0,
        }

    # Prioritize realistic operating budgets.
    weights = []
    for r in rows:
        b = int(r.get("max_tokens", 0))
        if b >= 90000:
            weights.append(0.45)
        elif b >= 65000:
            weights.append(0.35)
        else:
            weights.append(0.20)
    total_w = sum(weights) or 1.0
    weights = [w / total_w for w in weights]

    def wavg(vals: List[float]) -> float:
        return sum(v * w for v, w in zip(vals, weights))

    token_reduction = wavg([float(r.get("token_reduction_pct", 0.0)) / 100.0 for r in rows])
    keepfull_retention = wavg([
        float(r.get("quality_weak_labels", {}).get("keepfull_retention", 0.0)) for r in rows
    ])
    evict_capture = wavg([
        float(r.get("quality_weak_labels", {}).get("evict_capture", 0.0)) for r in rows
    ])
    needed_retention = wavg([
        float(r.get("quality_future_ref_proxy", {}).get("needed_retention", 0.0)) for r in rows
    ])
    return {
        "token_reduction": round(token_reduction, 4),
        "keepfull_retention": round(keepfull_retention, 4),
        "evict_capture": round(evict_capture, 4),
        "needed_retention": round(needed_retention, 4),
    }


def _composite_score(metrics: Dict[str, float]) -> float:
    # Higher is better. Heavily penalize loss of critical context.
    return round(
        (
            0.45 * metrics["keepfull_retention"]
            + 0.25 * metrics["needed_retention"]
            + 0.20 * metrics["evict_capture"]
            + 0.10 * metrics["token_reduction"]
        )
        * 100.0,
        3,
    )


def run_pack(
    dump_paths: Sequence[Path],
    budgets: Sequence[int],
    profile_names: Sequence[str] | None = None,
    future_lookahead: int = 30,
) -> Dict[str, Any]:
    names = list(profile_names) if profile_names else list(PROFILES.keys())
    rows: List[Dict[str, Any]] = []

    for profile_name in names:
        overrides = PROFILES.get(profile_name)
        if overrides is None:
            continue
        per_dump = {}
        agg_metrics = {
            "token_reduction": 0.0,
            "keepfull_retention": 0.0,
            "evict_capture": 0.0,
            "needed_retention": 0.0,
        }
        count = 0
        for p in dump_paths:
            r = run_benchmark(
                p,
                budgets=budgets,
                overrides=overrides,
                future_lookahead=future_lookahead,
            )
            per_dump[str(p)] = r
            m = _weighted_means(r)
            for k in agg_metrics:
                agg_metrics[k] += m[k]
            count += 1
        if count:
            for k in agg_metrics:
                agg_metrics[k] = round(agg_metrics[k] / count, 4)
        score = _composite_score(agg_metrics)
        rows.append(
            {
                "profile": profile_name,
                "overrides": overrides,
                "aggregate_metrics": agg_metrics,
                "composite_score": score,
                "per_dump": per_dump,
            }
        )

    rows.sort(key=lambda x: x["composite_score"], reverse=True)
    return {
        "dumps": [str(p) for p in dump_paths],
        "budgets": list(budgets),
        "future_lookahead": future_lookahead,
        "profiles_ranked": rows,
        "winner": rows[0]["profile"] if rows else None,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run context-management benchmark pack.")
    p.add_argument("--dump", action="append", required=True, help="Context dump path (repeatable)")
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    p.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES.keys()),
        help=f"Profiles to run. Available: {', '.join(PROFILES.keys())}",
    )
    p.add_argument("--future-lookahead", type=int, default=30)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dumps = [Path(p) for p in args.dump]
    result = run_pack(
        dump_paths=dumps,
        budgets=args.budgets,
        profile_names=args.profiles,
        future_lookahead=args.future_lookahead,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote benchmark pack: {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
