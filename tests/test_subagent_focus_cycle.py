"""Regression tests for sub-agent focus cycling (Ctrl+E / F4).

Cycle order: parent → agent1 → … → agentN → parent, with agents listed in
display order (focused first, then running, then completed/interrupted, then
error). Also covers /agent <n> index resolution logic (order parity).
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.config import Config
from harness.sub_agent_manager import SubAgentManager


def _make_manager(tmp_path):
    from pathlib import Path
    return SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / f"{name}.json",
    )


class _Task:
    def __init__(self, done):
        self._done = done

    def done(self):
        return self._done

    def cancel(self):
        pass


def _inst(name, status, task=None, pending=None, session_path=None):
    return SimpleNamespace(
        name=name,
        agent=None,
        task=task,
        status=status,
        output="",
        final_result="",
        result_log=[],
        turn_seq=0,
        signals=None,
        pending=pending,
        turn_started_at=None,
        tee=None,
        session_path=session_path,
        created_at=0.0,
        completed_at=None,
        completion_notified=False,
        last_error=None,
    )


# ── next_focus ordering ──────────────────────────────────────────────────────

def test_next_focus_empty_registry(tmp_path):
    manager = _make_manager(tmp_path)
    assert manager.next_focus(None) is None
    manager.cleanup()


def test_next_focus_cycles_parent_to_first_agent(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["alpha"] = _inst("alpha", "running", task=_Task(False), session_path=None)
    manager._agents["beta"] = _inst("beta", "completed", task=_Task(True), session_path=None)
    assert manager.next_focus(None) in ("alpha", "beta")  # display order
    manager.cleanup()


def test_next_focus_wraps_to_parent(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["alpha"] = _inst("alpha", "completed", task=_Task(True), session_path=None)
    # Parent (None) → alpha
    assert manager.next_focus(None) == "alpha"
    # alpha → parent (wrap)
    assert manager.next_focus("alpha") is None
    manager.cleanup()


def test_next_focus_running_first(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["done-agent"] = _inst("done-agent", "completed", task=_Task(True), session_path=None)
    manager._agents["busy-agent"] = _inst("busy-agent", "running", task=_Task(False), session_path=None)
    assert manager.next_focus(None) == "busy-agent"  # running ranks first
    manager.cleanup()


def test_next_focus_three_agent_cycle(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["a"] = _inst("a", "running", task=_Task(False), session_path=None)
    manager._agents["b"] = _inst("b", "running", task=_Task(False), session_path=None)
    manager._agents["c"] = _inst("c", "completed", task=_Task(True), session_path=None)
    # Full cycle: parent → a → b → c → parent
    assert manager.next_focus(None) == "a"
    assert manager.next_focus("a") == "b"
    assert manager.next_focus("b") == "c"
    assert manager.next_focus("c") is None
    manager.cleanup()


def test_next_focus_unknown_current_starts_at_first(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["a"] = _inst("a", "running", task=_Task(False), session_path=None)
    assert manager.next_focus("ghost") == "a"
    manager.cleanup()


# ── /agent index resolution parity ────────────────────────────────────────────

def test_ordered_names_matches_agents_sort_key():
    """The manager's display order must match the /agents table sort key
    (focused → running → completed/interrupted → error) so indices agree."""
    manager = SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=".",
        get_session_path_fn=lambda ws, name: None,
    )
    manager._agents["err-agent"] = _inst("err-agent", "error", task=_Task(True), session_path=None)
    manager._agents["done-agent"] = _inst("done-agent", "completed", task=_Task(True), session_path=None)
    manager._agents["busy-agent"] = _inst("busy-agent", "running", task=_Task(False), session_path=None)
    names = manager._ordered_names()
    assert names == ["busy-agent", "done-agent", "err-agent"]
    manager.cleanup()
