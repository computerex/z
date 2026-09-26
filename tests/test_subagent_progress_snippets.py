"""Regression tests: readable sub-agent progress snippets and batched spawns.

1. list_agents() snippets for RUNNING agents previously contained raw ANSI
   escape sequences — the sub-agent's thinking stream wrote ESC[2m + char +
   ESC[0m per character into the tee buffer, so progress was unreadable
   escape soup like "[0m[2mm[0m[2mo[0m...". Snippets are now ANSI-stripped,
   and the thinking stream batches its styling into one span per chunk.

2. get_agent_output() on a RUNNING agent previously returned only
   "still running" — it now returns a tail of the partial output.

3. Multiple create_agent calls in one assistant message previously executed
   only the first (1-tool-per-turn policy), blocking parallel dispatch.
   create_agent spawns are non-blocking, so all of them in a batch now
   execute.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.sub_agent_manager import SubAgentManager, TeeWriter, strip_ansi


# ── strip_ansi ─────────────────────────────────────────────────────────────

def test_strip_ansi_removes_escape_sequences():
    noisy = "\x1b[2mm\x1b[0m\x1b[2mo\x1b[0m\x1b[2mr\x1b[0m\x1b[2my\x1b[0m"
    assert strip_ansi(noisy) == "mory"


def test_strip_ansi_handles_csi_osc_and_blank_runs():
    text = "\x1b[1;32mOK\x1b[0m\n\n\n\n\x1b]0;title\x07plain"
    cleaned = strip_ansi(text)
    assert "OK" in cleaned and "plain" in cleaned
    assert "\x1b" not in cleaned
    assert "\n\n\n\n" not in cleaned  # blank runs collapsed


def test_strip_ansi_empty():
    assert strip_ansi("") == ""
    assert strip_ansi(None) == ""


# ── list() snippet stripping for running agents ─────────────────────────────

def _make_manager(tmp_path):
    from harness.config import Config
    return SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: os.path.join(ws, f"{name}.json"),
    )


def test_list_snippet_running_agent_is_plain_text(tmp_path):
    manager = _make_manager(tmp_path)
    tee = TeeWriter(io_stdout := None)  # placeholder, replaced below
    tee = TeeWriter(real_stdout=None)
    # Simulate the sub-agent's styled thinking output landing in the tee
    tee.write("\x1b[2mThinking:\x1b[0m\n")
    tee.write("\x1b[2maudit\x1b[0m\x1b[2m in\x1b[0m\x1b[2m progress\x1b[0m\n")
    inst = SimpleNamespace(
        name="runner",
        agent=None,
        task=None,
        status="running",
        output="",
        final_result="",
        tee=tee,
        session_path=None,
        created_at=__import__("time").time(),
        completed_at=None,
        completion_notified=False,
        last_error=None,
    )
    manager._agents["runner"] = inst

    listed = manager.list()
    assert len(listed) == 1
    snippet = listed[0]["output"]
    assert "\x1b" not in snippet, f"ANSI leaked into snippet: {snippet!r}"
    assert "audit in progress" in snippet
    manager.cleanup()


def test_list_snippet_completed_agent_prefers_final_result(tmp_path):
    manager = _make_manager(tmp_path)
    tee = TeeWriter(real_stdout=None)
    tee.write("\x1b[2mtranscript\x1b[0m\n")
    inst = SimpleNamespace(
        name="done",
        agent=None,
        task=None,
        status="completed",
        output=tee.getvalue(),
        final_result="VERDICT: clean",
        tee=tee,
        session_path=None,
        created_at=__import__("time").time(),
        completed_at=__import__("time").time(),
        completion_notified=True,
        last_error=None,
    )
    manager._agents["done"] = inst

    listed = manager.list()
    assert listed[0]["output"] == "VERDICT: clean"
    manager.cleanup()


# ── get_agent_output partial tail for running agents ────────────────────────

@pytest.mark.asyncio
async def test_get_agent_output_running_agent_returns_partial_tail():
    from harness.tools.subagent import get_agent_output

    tee = TeeWriter(real_stdout=None)
    tee.write("\x1b[2mworking on it\x1b[0m\nstill running...\n")

    class _Manager:
        def get(self, name):
            return SimpleNamespace(
                status="running",
                final_result="",
                output="",
                tee=tee,
                task=SimpleNamespace(done=lambda: False),
            )

    handlers = SimpleNamespace(sub_agent_manager=_Manager())
    result = await get_agent_output(handlers, {"name": "runner"})

    assert "still running" in result
    assert "working on it" in result
    assert "\x1b" not in result, f"ANSI leaked into partial output: {result!r}"


@pytest.mark.asyncio
async def test_get_agent_output_running_agent_without_output():
    from harness.tools.subagent import get_agent_output

    class _Manager:
        def get(self, name):
            return SimpleNamespace(
                status="running",
                final_result="",
                output="",
                tee=None,
                task=SimpleNamespace(done=lambda: False),
            )

    handlers = SimpleNamespace(sub_agent_manager=_Manager())
    result = await get_agent_output(handlers, {"name": "runner"})
    assert "still running" in result
    assert "no output yet" in result


# ── batched create_agent execution (parallel dispatch) ──────────────────────

def test_batch_create_agent_partitioning():
    """Static check: the 1-tool-per-turn policy partitions create_agent calls
    out of the ignored set so they execute alongside the first tool."""
    import inspect
    import harness.cline_agent as ca

    src = inspect.getsource(ca)
    # Main path partitions create_agent spawns
    assert 't for t in all_tool_calls[1:] if t.name == "create_agent"' in src
    assert 't for t in all_tool_calls[1:] if t.name != "create_agent"' in src
    # Pre-completion path partitions too
    assert 't for t in actionable_tools[1:] if t.name == "create_agent"' in src
    assert 't for t in actionable_tools[1:] if t.name != "create_agent"' in src
    # Spawn results are appended as real role="tool" messages (not synthetic)
    assert "Tool exec START (batch spawn)" in src
    assert "Tool exec START (pre-completion batch spawn)" in src


@pytest.mark.asyncio
async def test_batched_create_agent_calls_all_execute():
    """End-to-end: a manager records every create() — the cline_agent batch
    code calls _execute_tool for each spawn. Here we simulate the dispatch
    contract the batch code relies on: create_agent executes against the
    tool handlers' sub_agent_manager."""
    from harness.tools.subagent import create_agent

    created = []

    class _Manager:
        def create(self, name, task):
            created.append((name, task))

    handlers = SimpleNamespace(
        sub_agent_manager=_Manager(),
        console=SimpleNamespace(print=lambda *a, **k: None),
    )

    r1 = await create_agent(handlers, {"name": "a1", "task": "t1"})
    r2 = await create_agent(handlers, {"name": "a2", "task": "t2"})
    assert "Created sub-agent 'a1'" in r1
    assert "Created sub-agent 'a2'" in r2
    assert created == [("a1", "t1"), ("a2", "t2")]


# ── process-wide log id counter (no cmd_N.log collisions) ───────────────────

def test_cmd_log_paths_unique_across_instances(tmp_path):
    """Two ToolHandlers instances (parent + sub-agent) must never produce
    the same cmd_N.log path — per-instance counters previously made
    concurrent agents collide on the same file (PermissionError on
    Windows, interleaved output elsewhere)."""
    from harness.tools import ToolHandlers
    from harness.tools.mcp import _get_cmd_log_path, _get_bg_log_path

    def mk():
        return ToolHandlers(
            config=None,
            console=None,
            workspace_path=str(tmp_path),
            context=None,
            duplicate_detector=None,
        )

    h1, h2 = mk(), mk()
    paths = {
        _get_cmd_log_path(h1),
        _get_cmd_log_path(h2),
        _get_cmd_log_path(h1),
        _get_bg_log_path(h1, next_tool_id := __import__("harness.tools._base", fromlist=["next_tool_id"]).next_tool_id()),
        _get_bg_log_path(h2, __import__("harness.tools._base", fromlist=["next_tool_id"]).next_tool_id()),
    }
    # All five paths must be distinct
    assert len(paths) == 5, f"log path collision: {paths}"


def test_next_tool_id_thread_safe_unique():
    """Concurrent threads must receive unique ids (no duplicates)."""
    import threading
    from harness.tools._base import next_tool_id

    results = []
    lock = threading.Lock()

    def grab():
        ids = [next_tool_id() for _ in range(50)]
        with lock:
            results.extend(ids)

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 200
    assert len(set(results)) == 200, "duplicate tool ids across threads"
