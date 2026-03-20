"""Tests for context replay dataset extraction and feature building."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.context_replay import extract_nodes_from_dump, build_training_matrix, run_replay


def test_extract_nodes_from_dump(tmp_path: Path):
    dump = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "[GUIDANCE_NUDGE type=introspect severity=strong calls=9]\n[IMPORTANT: ...]"},
            {"role": "assistant", "content": "<read_file><path>a.py</path></read_file>"},
            {"role": "user", "content": "[execute_command result]\n$ pytest -q\nok"},
        ]
    }
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(dump), encoding="utf-8")
    nodes = extract_nodes_from_dump(p)
    assert len(nodes) == 4
    assert any(n.msg_type == "guidance_nudge" for n in nodes)
    assert any(n.msg_type == "command_output" for n in nodes)


def test_build_training_matrix_shapes(tmp_path: Path):
    dump = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "[GUIDANCE_NUDGE type=error-analysis severity=gentle]\n[Note: This result contains errors.]"},
            {"role": "assistant", "content": "Analysis " * 120},
            {"role": "user", "content": "[search_files result]\nsrc/a.py:1:TODO"},
        ]
    }
    p = tmp_path / "dump2.json"
    p.write_text(json.dumps(dump), encoding="utf-8")
    nodes = extract_nodes_from_dump(p)
    x, y, meta = build_training_matrix(nodes, use_embeddings=False)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(meta, dict)
    assert x.shape[0] == len(nodes)
    assert y.shape[0] == len(nodes)
    assert x.shape[1] >= 8
    assert meta["embedding_backend_used"] == "lexical_proxy"


def test_run_replay_with_hash_backend(tmp_path: Path):
    dump = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Please inspect src/app.py and fix failing tests."},
            {"role": "assistant", "content": "<execute_command><command>pytest -q</command></execute_command>"},
            {"role": "user", "content": "[execute_command result]\nFAILED tests/test_app.py::test_x"},
        ]
    }
    p = tmp_path / "dump3.json"
    p.write_text(json.dumps(dump), encoding="utf-8")
    out = run_replay([p], train=True, embedding_backend="hash")
    assert out["embedding_backend_requested"] == "hash"
    assert out["total_nodes"] == 4
    assert "training" in out
