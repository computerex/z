"""Regression tests for the non-blocking sub-agent turn queue.

The old REPL focused-agent path blocked in loop.run_until_complete(run(...)),
and run() silently awaited the in-flight task first — no prompt, no status,
no cancel: a silent command looked like a frozen terminal. Now:

- queue_input (REPL path): idle → start immediately; busy → queue + ack,
  prompt returns instantly, queued turn output streams via the TeeWriter.
- Drain rules: consecutive user entries coalesce into one turn; each model
  entry (send_agent_input) gets its own turn with its future resolved.
- pause/delete flush the queue and resolve waiting futures.
- send_agent_input on a busy agent returns a queued-ack by default (wait=true
  restores blocking).
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

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


class FakeAgent:
    """Agent whose turns block until released, recording every turn text."""

    def __init__(self):
        self.turns = []
        self._gates = []

    async def run_message(self, text, enable_interrupt=True):
        self.turns.append(text)
        gate = asyncio.Event()
        self._gates.append(gate)
        await gate.wait()
        return f"reply to: {text[:20]}"

    def release(self, idx=None):
        if idx is None and self._gates:
            idx = len(self._gates) - 1
        if self._gates and idx is not None and idx < len(self._gates):
            self._gates[idx].set()


def _instance(name, agent):
    return SimpleNamespace(
        name=name,
        agent=agent,
        task=None,
        status="created",
        output="",
        final_result="",
        result_log=[],
        turn_seq=0,
        signals=None,
        pending=None,
        turn_started_at=None,
        tee=None,
        session_path=None,
        created_at=0.0,
        completed_at=None,
        completion_notified=False,
        last_error=None,
    )


# ── queue_input (REPL path) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_queue_input_idle_starts_immediately(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    result = await manager.queue_input("audit", "hello")
    assert result == "started"
    # Give the scheduled task a chance to run
    await asyncio.sleep(0.05)
    assert agent.turns == ["hello"]
    assert inst.status == "running"
    agent.release()
    await asyncio.sleep(0.05)
    manager.cleanup()


@pytest.mark.asyncio
async def test_queue_input_busy_queues_and_returns_instantly(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)  # first turn now blocking in FakeAgent
    result = await manager.queue_input("audit", "second")
    assert result == "queued"
    assert len(inst.pending) == 1
    # The prompt returns instantly; first turn is still running
    assert agent.turns == ["first"]

    agent.release()
    await asyncio.sleep(0.1)
    # Drainer picked up the queued input
    assert agent.turns == ["first", "second"]
    agent.release()
    await asyncio.sleep(0.05)
    manager.cleanup()


@pytest.mark.asyncio
async def test_queue_input_drops_self_notification(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    inst.status = "completed"
    manager._agents["audit"] = inst

    notif = "[SYSTEM: Sub-agent 'audit' has completed its task. ...]"
    result = await manager.queue_input("audit", notif)
    assert result == "dropped"
    assert agent.turns == []
    assert inst.pending is None or inst.pending == []
    manager.cleanup()


@pytest.mark.asyncio
async def test_user_entries_coalesce_into_one_turn(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)
    await manager.queue_input("audit", "second")
    await manager.queue_input("audit", "third")
    assert len(inst.pending) == 2

    agent.release()  # first turn completes → drainer coalesces both user entries
    await asyncio.sleep(0.1)
    assert agent.turns == ["first", "second\n\nthird"]
    agent.release()
    await asyncio.sleep(0.05)
    assert inst.pending == []
    manager.cleanup()


# ── run() (model path via send_agent_input) ────────────────────────────────

@pytest.mark.asyncio
async def test_run_idle_agent_returns_reply(tmp_path):
    manager = _make_manager(tmp_path)
    class FastAgent:
        def __init__(self):
            self.turns = []
        async def run_message(self, text, enable_interrupt=True):
            self.turns.append(text)
            return "the reply"
    agent = FastAgent()
    inst = _instance("audit", agent)
    inst.status = "completed"  # idle
    manager._agents["audit"] = inst

    result = await manager.run("audit", "continue the analysis")
    assert result == "the reply"
    assert agent.turns == ["continue the analysis"]
    manager.cleanup()


@pytest.mark.asyncio
async def test_run_busy_agent_queues_with_ack_by_default(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)

    result = await manager.run("audit", "check the SQL too", wait=False)
    assert result.startswith("Queued (position 1)")
    assert "get_agent_output" in result

    agent.release()
    await asyncio.sleep(0.1)
    # Model entry got its OWN turn (not coalesced with anything)
    assert agent.turns == ["first", "check the SQL too"]
    agent.release()
    await asyncio.sleep(0.05)
    manager.cleanup()


@pytest.mark.asyncio
async def test_run_busy_agent_wait_true_returns_turn_result(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)

    run_task = asyncio.ensure_future(
        manager.run("audit", "check the SQL too", wait=True)
    )
    await asyncio.sleep(0.05)
    agent.release()  # first turn completes → drainer runs the queued model entry
    await asyncio.sleep(0.05)
    agent.release()  # release the model entry's turn
    result = await asyncio.wait_for(run_task, timeout=2.0)
    assert result == "reply to: check the SQL too"
    manager.cleanup()


# ── pause/delete flush ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pause_flushes_queue_and_resolves_waiters(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)
    run_task = asyncio.ensure_future(
        manager.run("audit", "queued model input", wait=True)
    )
    await asyncio.sleep(0.05)

    # Pause: task cancelled, queue flushed, waiter resolved (not hung)
    manager.pause("audit")
    result = await asyncio.wait_for(run_task, timeout=1.0)
    assert result == "[Sub-agent paused]"
    assert inst.pending == []
    manager.cleanup()


@pytest.mark.asyncio
async def test_delete_flushes_queue(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)
    run_task = asyncio.ensure_future(
        manager.run("audit", "queued model input", wait=True)
    )
    await asyncio.sleep(0.05)

    assert manager.delete("audit") is True
    result = await asyncio.wait_for(run_task, timeout=1.0)
    assert result == "[Sub-agent deleted]"
    manager.cleanup()


# ── unfocus-mid-queue: draining is focus-independent ────────────────────────

@pytest.mark.asyncio
async def test_queue_drains_when_unfocused(tmp_path):
    manager = _make_manager(tmp_path)
    agent = FakeAgent()
    inst = _instance("audit", agent)
    manager._agents["audit"] = inst

    await manager.queue_input("audit", "first")
    await asyncio.sleep(0.05)
    await manager.queue_input("audit", "second")
    # "Unfocus" — irrelevant to the drainer
    manager.set_focused(None)

    agent.release()
    await asyncio.sleep(0.1)
    assert agent.turns == ["first", "second"]
    agent.release()
    await asyncio.sleep(0.05)
    assert inst.status == "completed"
    manager.cleanup()


# ── send_agent_input tool level ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_agent_input_tool_passes_wait_param(tmp_path):
    from harness.tools.subagent import send_agent_input

    class _Manager:
        def __init__(self):
            self.calls = []

        def get(self, name):
            if name == "audit":
                return _instance("audit", FakeAgent())
            return None

        async def run(self, name, text, wait=True):
            self.calls.append((name, text, wait))
            return "ack"

    class _FakeConsole:
        def print(self, *a, **k):
            pass

    handlers = SimpleNamespace(
        sub_agent_manager=_Manager(),
        console=_FakeConsole(),
    )
    result = await send_agent_input(handlers, {"name": "audit", "input": "go"})
    assert result == "ack"
    assert handlers.sub_agent_manager.calls == [("audit", "go", False)]

    result = await send_agent_input(
        handlers, {"name": "audit", "input": "go", "wait": "true"}
    )
    assert handlers.sub_agent_manager.calls[-1] == ("audit", "go", True)
