"""Offline replay and classifier training for semantic context management.

This module turns harness context dumps into structured samples, builds
embedding + metadata features, and trains fast classifier ensembles.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .context_management import estimate_tokens
from .smart_context import SmartContextManager
from .todo_manager import TodoManager

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

ACTIONS = ["KEEP_FULL", "SUMMARIZE", "ARCHIVE_WITH_BREADCRUMB", "EVICT"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

AUTO_BACKEND = "hash_char_wb"


@dataclass
class ReplayNode:
    dump_path: str
    index: int
    role: str
    msg_type: str
    source: str
    tokens: int
    age_from_end: int
    content: str
    weak_label: str


def _load_dump(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _weak_label(msg_type: str, tokens: int, role: str) -> str:
    """Weak supervision labels to bootstrap classifier training."""
    if msg_type == "guidance_nudge":
        return "EVICT"
    if msg_type in ("todo_result", "context_result"):
        return "EVICT"
    if msg_type in ("command_output", "search_result", "other_tool_result"):
        return "ARCHIVE_WITH_BREADCRUMB" if tokens >= 120 else "SUMMARIZE"
    if msg_type == "assistant_analysis":
        return "SUMMARIZE" if tokens >= 160 else "KEEP_FULL"
    if role == "system":
        return "KEEP_FULL"
    return "KEEP_FULL"


def extract_nodes_from_dump(path: Path) -> List[ReplayNode]:
    data = _load_dump(path)
    raw_messages = data.get("messages", [])
    scm = SmartContextManager(TodoManager())
    nodes: List[ReplayNode] = []
    total = len(raw_messages)
    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        text = _extract_text(msg.get("content", ""))
        if not text:
            continue
        msg_type, source = scm._classify(text, role)
        tok = estimate_tokens(text)
        label = _weak_label(msg_type, tok, role)
        nodes.append(
            ReplayNode(
                dump_path=str(path),
                index=i,
                role=role,
                msg_type=msg_type,
                source=source,
                tokens=tok,
                age_from_end=max(0, (total - 1) - i),
                content=text,
                weak_label=label,
            )
        )
    return nodes


def _metadata_features(nodes: Sequence[ReplayNode]) -> np.ndarray:
    role_map = {"system": 0, "user": 1, "assistant": 2}
    msg_type_vocab = {
        "file_read": 0,
        "command_output": 1,
        "search_result": 2,
        "assistant_analysis": 3,
        "assistant_tool_call": 4,
        "todo_result": 5,
        "context_result": 6,
        "other_tool_result": 7,
        "guidance_nudge": 8,
        "other": 9,
    }
    rows: List[List[float]] = []
    for n in nodes:
        rows.append(
            [
                float(role_map.get(n.role, 3)),
                float(msg_type_vocab.get(n.msg_type, 10)),
                float(n.tokens),
                float(n.age_from_end),
                1.0 if n.msg_type == "guidance_nudge" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _lexical_proxy_features(nodes: Sequence[ReplayNode]) -> np.ndarray:
    rows: List[List[float]] = []
    for n in nodes:
        lower = n.content.lower()
        rows.append(
            [
                float(lower.count("error")),
                float(lower.count("introspect")),
                float(lower.count("todo")),
                float(lower.count("<read_file>")),
                float(lower.count("<execute_command>")),
                float(len(n.content) / 1000.0),
                1.0 if "result" in lower else 0.0,
                1.0 if "guidance_nudge" in lower else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _hash_embedding_features(nodes: Sequence[ReplayNode], n_features: int = 256) -> np.ndarray:
    if not HAS_SKLEARN:
        return _lexical_proxy_features(nodes)
    texts = [n.content[:4000] for n in nodes]
    vect = HashingVectorizer(
        n_features=n_features,
        analyzer="char_wb",
        ngram_range=(3, 5),
        alternate_sign=False,
        norm="l2",
    )
    mat = vect.transform(texts)
    return mat.toarray().astype(np.float32)


def _embedding_features(
    nodes: Sequence[ReplayNode],
    backend: str = AUTO_BACKEND,
) -> Tuple[np.ndarray, str]:
    """Compute hash-based embedding features (semantic backends removed)."""
    if not nodes:
        return np.zeros((0, 8), dtype=np.float32), "none"
    return _hash_embedding_features(nodes), "hash_char_wb"


def build_training_matrix(
    nodes: Sequence[ReplayNode],
    use_embeddings: bool = True,
    embedding_backend: str = AUTO_BACKEND,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x_meta = _metadata_features(nodes)
    backend_used = "none"
    if use_embeddings:
        x_emb, backend_used = _embedding_features(nodes, backend=embedding_backend)
    else:
        x_emb = _lexical_proxy_features(nodes)
        backend_used = "lexical_proxy"
    x = np.concatenate([x_meta, x_emb], axis=1)
    y = np.asarray([ACTION_TO_ID[n.weak_label] for n in nodes], dtype=np.int32)
    meta = {
        "embedding_backend_requested": embedding_backend,
        "embedding_backend_used": backend_used,
        "meta_features": int(x_meta.shape[1]),
        "embedding_features": int(x_emb.shape[1]),
    }
    return x, y, meta


def _safe_train_test_split(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 4:
        return x, x, y, y
    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
    return train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)


def _model_predict_proba(model: Any, x: np.ndarray, total_classes: int) -> np.ndarray:
    classes = getattr(model, "classes_", None)

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x)
        if isinstance(score, list):
            score = np.asarray(score)
        score = np.asarray(score, dtype=np.float32)
        if score.ndim == 1:
            score = np.stack([-score, score], axis=1)
        score = score - np.max(score, axis=1, keepdims=True)
        exp = np.exp(score)
        denom = np.clip(exp.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
        raw = exp / denom
    else:
        pred = model.predict(x)
        n_classes = int(pred.max()) + 1 if len(pred) else 1
        raw = np.zeros((len(pred), n_classes), dtype=np.float32)
        for i, cls in enumerate(pred):
            raw[i, int(cls)] = 1.0

    if classes is None:
        return raw

    out = np.zeros((raw.shape[0], total_classes), dtype=np.float32)
    class_ids = [int(c) for c in classes]
    for j, cls_id in enumerate(class_ids):
        if 0 <= cls_id < total_classes:
            out[:, cls_id] = raw[:, j]
    return out


def train_fast_ensemble(
    nodes: Sequence[ReplayNode],
    embedding_backend: str = AUTO_BACKEND,
) -> Dict[str, Any]:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn not installed; cannot train classifiers.")
    if not nodes:
        raise RuntimeError("No nodes to train on.")

    x, y, build_meta = build_training_matrix(nodes, use_embeddings=True, embedding_backend=embedding_backend)
    x_train, x_test, y_train, y_test = _safe_train_test_split(x, y)
    unique, counts = np.unique(y_train, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0

    models: Dict[str, Any] = {}
    reports: Dict[str, Any] = {}
    probas: List[np.ndarray] = []

    lr = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=None)),
        ]
    )
    lr.fit(x_train, y_train)
    models["logistic"] = lr
    y_lr = lr.predict(x_test)
    reports["logistic_report"] = classification_report(y_test, y_lr, output_dict=True, zero_division=0)
    probas.append(_model_predict_proba(lr, x_test, total_classes=len(ACTIONS)))

    sgd = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-3)),
        ]
    )
    sgd.fit(x_train, y_train)
    models["sgd_log"] = sgd
    y_sgd = sgd.predict(x_test)
    reports["sgd_report"] = classification_report(y_test, y_sgd, output_dict=True, zero_division=0)
    probas.append(_model_predict_proba(sgd, x_test, total_classes=len(ACTIONS)))

    rf = RandomForestClassifier(
        n_estimators=160,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(x_train, y_train)
    models["random_forest"] = rf
    y_rf = rf.predict(x_test)
    reports["random_forest_report"] = classification_report(y_test, y_rf, output_dict=True, zero_division=0)
    probas.append(_model_predict_proba(rf, x_test, total_classes=len(ACTIONS)))

    if len(unique) >= 2 and min_class >= 2:
        cv = 3 if min_class >= 3 else 2
        svc_base = Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
            ]
        )
        svc_model: Any = CalibratedClassifierCV(svc_base, method="sigmoid", cv=cv)
        svc_model.fit(x_train, y_train)
        models["linearsvc_calibrated"] = svc_model
        y_svc = svc_model.predict(x_test)
        reports["linearsvc_report"] = classification_report(y_test, y_svc, output_dict=True, zero_division=0)
        probas.append(_model_predict_proba(svc_model, x_test, total_classes=len(ACTIONS)))
    else:
        reports["linearsvc_report"] = {"note": "LinearSVC calibration skipped due to insufficient class support."}

    # Soft-vote ensemble over all available probabilistic models.
    proba_ens = (
        np.mean(probas, axis=0)
        if probas
        else _model_predict_proba(lr, x_test, total_classes=len(ACTIONS))
    )
    y_ens = proba_ens.argmax(axis=1)

    reports["ensemble_report"] = classification_report(y_test, y_ens, output_dict=True, zero_division=0)

    out = {
        "samples": len(nodes),
        "features": int(x.shape[1]),
        "class_distribution": {a: int((y == i).sum()) for a, i in ACTION_TO_ID.items()},
        "models": list(models.keys()),
        "build": build_meta,
        **reports,
    }
    return out


def run_replay(
    dump_paths: Sequence[Path],
    train: bool = True,
    embedding_backend: str = AUTO_BACKEND,
) -> Dict[str, Any]:
    nodes: List[ReplayNode] = []
    per_dump: Dict[str, Dict[str, Any]] = {}
    for path in dump_paths:
        ns = extract_nodes_from_dump(path)
        nodes.extend(ns)
        per_dump[str(path)] = {
            "nodes": len(ns),
            "guidance_nodes": sum(1 for n in ns if n.msg_type == "guidance_nudge"),
            "avg_tokens": float(np.mean([n.tokens for n in ns])) if ns else 0.0,
        }

    out: Dict[str, Any] = {
        "dumps": per_dump,
        "total_nodes": len(nodes),
        "embedding_backend_requested": embedding_backend,
    }
    if train:
        out["training"] = train_fast_ensemble(nodes, embedding_backend=embedding_backend)
    return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay context dumps and train fast classifier ensembles.")
    p.add_argument("--dump", action="append", required=True, help="Path to a context dump JSON (repeatable).")
    p.add_argument("--no-train", action="store_true", help="Only extract/analyze nodes; skip model training.")
    p.add_argument(
        "--embed-backend",
        default=AUTO_BACKEND,
        help=(
            "Embedding backend: hash (default). "
            "Semantic/sentence-transformer backends have been removed."
        ),
    )
    p.add_argument("--out", default="", help="Optional JSON output path.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    paths = [Path(p) for p in args.dump]
    result = run_replay(
        paths,
        train=not args.no_train,
        embedding_backend=args.embed_backend,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote replay report: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
