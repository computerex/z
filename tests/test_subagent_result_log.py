"""Regression tests for the per-agent result log (consume-on-fetch).

The old single-`final_result` field was overwritten on every turn: if turn 2
completed before turn 1's notification was consumed (parent busy in a long
turn), turn 1's result was silently lost to the ANSI transcript. The result
log keeps every turn's result until fetched.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.sub_agent_manager import SubAgentManager, TurnResult


def _make_manager(tmp_path):
    from pathlib import Path
    from harness.config import Config
    return SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / f"{name}.json",
    )


def _instance(name="audit"):
    return SimpleNamespace(
        name=name,
        agent=None,
        task=None,
        status="completed",
        output="",
        final_result="",
        result_log=[],
        turn_seq=0,
        tee=None,
        session_path=None,
        created_at=0.0,
        completed_at=None,
        completion_notified=False,
        last_error=None,
    )


# ── Recording ───────────────────────────────────────────────────────────────

def test_record_result_appends_monotonically(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _instance()
    manager._agents["audit"] = inst

    manager._record_result(inst, "completed", "first turn result")
    manager._record_result(inst, "completed", "second turn result")
    manager._record_result(inst, "error", "[Sub-agent error: boom]", error="boom")

    assert [r.turn_id for r in inst.result_log] == [1, 2, 3]
    assert inst.result_log[0].text == "first turn result"
    assert inst.result_log[2].status == "error"
    assert inst.result_log[2].error == "boom"
    manager.cleanup()


def test_record_result_defensive_against_foreign_instances(tmp_path):
    """Tests/embedders may pass SimpleNamespace without result_log — recording
    must not crash."""
    manager = _make_manager(tmp_path)
    inst = SimpleNamespace(name="x")  # no result_log attr
    manager._record_result(inst, "completed", "ok")
    assert inst.turn_seq == 1
    assert inst.result_log[0].text == "ok"
    manager.cleanup()


# ── Consume-on-fetch ────────────────────────────────────────────────────────

def test_fetch_new_results_returns_and_clears(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _instance()
    manager._agents["audit"] = inst
    manager._record_result(inst, "completed", "r1")
    manager._record_result(inst, "completed", "r2")

    got = manager.fetch_new_results("audit")
    assert [r.text for r in got] == ["r1", "r2"]
    # Second fetch: nothing until a new turn completes
    assert manager.fetch_new_results("audit") == []
    manager.cleanup()


def test_fetch_new_results_on_unknown_agent_raises_keyerror(tmp_path):
    manager = _make_manager(tmp_path)
    with pytest.raises(KeyError):
        manager.fetch_new_results("ghost")
    manager.cleanup()


def test_overlapping_turns_both_retrievable(tmp_path):
    """The exact regression: turn 2 completes before turn 1's result was
    fetched — both must be retrievable from the log."""
    manager = _make_manager(tmp_path)
    inst = _instance()
    manager._agents["audit"] = inst

    # Turn 1 completes, notification fires but is NOT yet acted on
    manager._record_result(inst, "completed", "VERDICT turn 1")
    # Turn 2 completes before the parent fetched turn 1's result
    manager._record_result(inst, "completed", "VERDICT turn 2")

    got = manager.fetch_new_results("audit")
    assert len(got) == 2
    assert "turn 1" in got[0].text
    assert "turn 2" in got[1].text
    manager.cleanup()


def test_fetch_defensive_when_log_missing(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _instance("x")
    inst.result_log = []
    manager._agents["x"] = inst
    assert manager.fetch_new_results("x") == []
    manager.cleanup()


# ── get_agent_output consume semantics (tool level) ─────────────────────────

@pytest.mark.asyncio
async def test_get_agent_output_drains_new_results(tmp_path):
    from harness.tools.subagent import get_agent_output

    inst = _instance()
    inst.status = "completed"
    inst.final_result = "stale latest"
    manager = _make_manager(tmp_path)
    manager._record_result(inst, "completed", "fresh turn 1 result")
    manager._agents["audit"] = inst

    handlers = SimpleNamespace(sub_agent_manager=manager)
    result = await get_agent_output(handlers, {"name": "audit"})

    assert "fresh turn 1 result" in result
    # Consumed: a second fetch falls back to the latest cached result
    result2 = await get_agent_output(handlers, {"name": "audit"})
    assert result2 == "stale latest"
    manager.cleanup()


@pytest.mark.asyncio
async def test_get_agent_output_multiple_new_results_labeled(tmp_path):
    from harness.tools.subagent import get_agent_output

    inst = _instance()
    manager = _make_manager(tmp_path)
    manager._record_result(inst, "completed", "result one")
    manager._record_result(inst, "completed", "result two")
    manager._agents["audit"] = inst

    handlers = SimpleNamespace(sub_agent_manager=manager)
    result = await get_agent_output(handlers, {"name": "audit"})

    assert "[Turn 1 result]:" in result
    assert "[Turn 2 result]:" in result
    assert "result one" in result and "result two" in result
    manager.cleanup()


@pytest.mark.asyncio
async def test_get_agent_output_error_result_labeled(tmp_path):
    from harness.tools.subagent import get_agent_output

    inst = _instance()
    manager = _make_manager(tmp_path)
    manager._record_result(inst, "error", "[Sub-agent error: API down]", error="API down")
    manager._agents["audit"] = inst

    handlers = SimpleNamespace(sub_agent_manager=manager)
    result = await get_agent_output(handlers, {"name": "audit"})

    assert "FAILED" in result
    assert "API down" in result
    manager.cleanup()


# ── list() exposes new_results count ────────────────────────────────────────

def test_list_includes_new_results_count(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _instance()
    inst.completed_at = 1.0  # list() freezes elapsed at completion for done agents
    manager._record_result(inst, "completed", "r1")
    manager._agents["audit"] = inst

    info = manager.list()[0]
    assert info["new_results"] == 1
    manager.cleanup()