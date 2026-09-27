"""Regression tests for per-agent keyboard signals (focused sub-agents).

Previously sub-agents ran with enable_interrupt=False and could NEVER observe
Esc (kill command) or Ctrl+B (background command) — the isolation design
conflated "unobserved background agent" with "agent the user is watching".
Now focused sub-agents observe their own per-agent signal object
(consume-on-read, per-turn scoping), while background agents remain isolated.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.sub_agent_manager import AgentSignals, SubAgentManager


def _make_manager(tmp_path):
    from pathlib import Path
    from harness.config import Config
    return SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / f"{name}.json",
    )


# ── AgentSignals primitive ───────────────────────────────────────────────────

def test_signals_consume_on_read():
    s = AgentSignals()
    assert s.consume_interrupt() is False
    s.interrupt = True
    assert s.consume_interrupt() is True
    assert s.consume_interrupt() is False  # consumed exactly once

    s.background = True
    assert s.consume_background() is True
    assert s.consume_background() is False


def test_signals_clear():
    s = AgentSignals()
    s.interrupt = True
    s.background = True
    s.clear()
    assert s.interrupt is False
    assert s.background is False


# ── ClineAgent signal consumption ──────────────────────────────────────────

def test_agent_consumes_own_signal():
    from harness.cline_agent import ClineAgent
    from harness.config import Config

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    # Parent: no agent_signals
    assert agent.agent_signals is None
    assert agent.consumes_own_signal("interrupt") is False

    # Focused sub-agent: signal set → consumed once
    agent.agent_signals = AgentSignals()
    agent.agent_signals.interrupt = True
    assert agent.consumes_own_signal("interrupt") is True
    assert agent.consumes_own_signal("interrupt") is False


def test_turn_interrupted_uses_own_signal():
    from harness.cline_agent import ClineAgent
    from harness.config import Config
    from harness import interrupt as interrupt_mod

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent._interrupt_enabled = False  # background sub-agent
    # Global interrupt set — agent must NOT observe it
    interrupt_mod._interrupt_state.interrupted = True
    try:
        assert agent._turn_interrupted() is False
        # But its own signal must be observed
        agent.agent_signals = AgentSignals()
        agent.agent_signals.interrupt = True
        assert agent._turn_interrupted() is True
        assert agent._turn_interrupted() is False  # consumed
    finally:
        interrupt_mod.reset_interrupt()


def test_turn_interrupted_parent_uses_global():
    from harness.cline_agent import ClineAgent
    from harness.config import Config
    from harness import interrupt as interrupt_mod

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent._interrupt_enabled = True
    agent.agent_signals = None  # parent
    interrupt_mod._interrupt_state.interrupted = True
    try:
        assert agent._turn_interrupted() is True
    finally:
        interrupt_mod.reset_interrupt()


# ── Shell tool signal consumption ────────────────────────────────────────────

def test_shell_own_signal_consumes():
    from harness.tools.shell import _own_signal

    class Owner:
        """Mirrors ClineAgent's signal interface."""

        def __init__(self):
            self.agent_signals = AgentSignals()

        def consumes_own_signal(self, kind):
            if kind == "interrupt":
                return self.agent_signals.consume_interrupt()
            if kind == "background":
                return self.agent_signals.consume_background()
            return False

    handlers = SimpleNamespace(owner_agent=Owner())
    handlers.owner_agent.agent_signals.interrupt = True
    assert _own_signal(handlers, "interrupt") is True
    assert _own_signal(handlers, "interrupt") is False  # consumed

    # No owner → False
    assert _own_signal(SimpleNamespace(), "interrupt") is False
    # Owner without signals attr → False (parent)
    assert _own_signal(SimpleNamespace(owner_agent=SimpleNamespace()), "interrupt") is False


# ── Manager.signal() setter + active_count ───────────────────────────────────

def test_manager_signal_sets_focused_agent(tmp_path):
    manager = _make_manager(tmp_path)
    inst = SimpleNamespace(
        name="audit",
        task=None,
        status="running",
        signals=AgentSignals(),
        pending=None,
        session_path=None,
    )
    manager._agents["audit"] = inst

    assert manager.signal("audit", "background") is True
    assert inst.signals.background is True
    assert manager.signal("audit", "interrupt") is True
    assert inst.signals.interrupt is True
    # Unknown agent → False
    assert manager.signal("ghost", "interrupt") is False
    manager.cleanup()


def test_manager_active_counts_running_tasks(tmp_path):
    import asyncio

    class RunningTask:
        def done(self):
            return False

        def cancel(self):
            pass

    class DoneTask:
        def done(self):
            return True

        def cancel(self):
            pass

    manager = _make_manager(tmp_path)
    manager._agents["a"] = SimpleNamespace(name="a", task=RunningTask(), pending=None, session_path=None, status="running")
    manager._agents["b"] = SimpleNamespace(name="b", task=DoneTask(), pending=None, session_path=None, status="completed")
    manager._agents["c"] = SimpleNamespace(name="c", task=DoneTask(), pending=["queued"], session_path=None, status="completed")
    assert manager.active_count() == 2  # a (running) + c (queued)
    manager.cleanup()

# ── Turn-end signal clearing ─────────────────────────────────────────────────

def test_run_agent_task_clears_signals_on_completion(tmp_path):
    import asyncio

    class OkAgent:
        async def run_message(self, text, enable_interrupt=True):
            return "done"

    manager = _make_manager(tmp_path)

    async def scenario():
        inst = SimpleNamespace(
            name="audit",
            agent=OkAgent(),
            task=None,
            status="running",
            output="",
            final_result="",
            result_log=[],
            turn_seq=0,
            signals=AgentSignals(),
            tee=None,
            session_path=None,
            completion_notified=False,
            last_error=None,
        )
        manager._agents["audit"] = inst
        # Stale signal set during the turn but never consumed
        inst.signals.interrupt = True
        await manager._run_agent_task(inst, "go")
        return inst

    inst = asyncio.run(scenario())
    assert inst.signals.interrupt is False  # cleared at turn end
    manager.cleanup()


def test_run_agent_task_clears_signals_on_error(tmp_path):
    import asyncio

    class ExplodingAgent:
        async def run_message(self, text, enable_interrupt=True):
            raise RuntimeError("boom")

    manager = _make_manager(tmp_path)

    async def scenario():
        inst = SimpleNamespace(
            name="audit",
            agent=ExplodingAgent(),
            task=None,
            status="running",
            output="",
            final_result="",
            result_log=[],
            turn_seq=0,
            signals=AgentSignals(),
            tee=None,
            session_path=None,
            completion_notified=False,
            last_error=None,
        )
        manager._agents["audit"] = inst
        inst.signals.background = True
        await manager._run_agent_task(inst, "go")
        return inst

    inst = asyncio.run(scenario())
    assert inst.signals.background is False  # cleared at turn end
    assert inst.status == "error"
    manager.cleanup()
