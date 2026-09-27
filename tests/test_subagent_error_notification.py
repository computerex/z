"""Regression tests: errored sub-agents must notify the parent.

The old check_completed()/peek_completed() matched only status == "completed" —
an errored sub-agent NEVER notified anyone, and the parent could wait forever
on a zombie agent unless it happened to poll list_agents().
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


def _errored_instance(name="audit"):
    return SimpleNamespace(
        name=name,
        agent=None,
        task=None,
        status="error",
        output="",
        final_result="",
        result_log=[],
        turn_seq=0,
        tee=None,
        session_path=None,
        completion_notified=False,
        last_error="API connection reset",
    )


# ── check_completed / peek_completed match errors ───────────────────────────

def test_check_completed_matches_errored_agent(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = _errored_instance()
    assert manager.check_completed() == "audit"
    # Consumed exactly once
    assert manager.check_completed() is None
    manager.cleanup()


def test_peek_completed_matches_errored_agent(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = _errored_instance()
    assert manager.peek_completed() == "audit"
    assert manager.peek_completed() == "audit"  # non-consuming
    manager.cleanup()


def test_errored_agent_notification_cycle_resets_on_new_turn(tmp_path):
    """After the error is notified, a subsequent turn must be able to notify
    again (completion_notified is reset by _run_agent_task)."""
    manager = _make_manager(tmp_path)
    inst = _errored_instance()
    manager._agents["audit"] = inst
    assert manager.check_completed() == "audit"
    # Simulate what _run_agent_task does at the start of a new result cycle
    inst.completion_notified = False
    assert manager.check_completed() == "audit"
    manager.cleanup()


# ── notification text ────────────────────────────────────────────────────────

def test_notification_text_for_error_embeds_last_error(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = _errored_instance()
    text = manager.notification_text("audit")
    assert "FAILED" in text
    assert "API connection reset" in text
    assert "get_agent_output(name='audit')" in text
    assert text.startswith("[SYSTEM:")
    assert text.endswith("]")
    manager.cleanup()


def test_notification_text_for_completed_unchanged(tmp_path):
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = SimpleNamespace(
        name="audit",
        agent=None,
        task=None,
        status="completed",
        output="",
        final_result="done",
        result_log=[],
        turn_seq=0,
        tee=None,
        session_path=None,
        completion_notified=False,
        last_error=None,
    )
    text = manager.notification_text("audit")
    assert "completed its task" in text
    assert "get_agent_output(name='audit')" in text
    assert "FAILED" not in text
    manager.cleanup()


# ── self-notification refusal covers errored agents ─────────────────────────

def test_run_refuses_own_error_notification(tmp_path):
    """Feeding an errored agent its own FAILED notification must not start a
    new turn — that would loop error→notify→run forever."""
    import asyncio

    calls = []

    class FakeAgent:
        async def run_message(self, text, enable_interrupt=True):
            calls.append(text)
            return "should not happen"

    inst = _errored_instance()
    inst.agent = FakeAgent()
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = inst

    notif = (
        "[SYSTEM: Sub-agent 'audit' has FAILED with error: API connection reset "
        "Use get_agent_output(name='audit') to see details.]"
    )
    result = asyncio.run(manager.run("audit", notif))

    assert calls == []  # NO new turn started
    assert result == ""  # final_result/output both empty on the fake instance
    manager.cleanup()


# ── error result recorded in result log ──────────────────────────────────────

def test_run_agent_task_records_error_result(tmp_path):
    import asyncio

    class ExplodingAgent:
        async def run_message(self, text, enable_interrupt=True):
            raise RuntimeError("provider exploded")

    manager = _make_manager(tmp_path)

    async def scenario():
        # Build a minimal instance through the manager's own task runner
        inst = SimpleNamespace(
            name="bomb",
            agent=ExplodingAgent(),
            task=None,
            status="running",
            output="",
            final_result="",
            result_log=[],
            turn_seq=0,
            tee=None,
            session_path=None,
            completion_notified=False,
            last_error=None,
        )
        manager._agents["bomb"] = inst
        result = await manager._run_agent_task(inst, "go")
        return inst, result

    inst, result = asyncio.run(scenario())
    assert "provider exploded" in result
    assert inst.status == "error"
    assert inst.last_error == "provider exploded"
    assert len(inst.result_log) == 1
    assert inst.result_log[0].status == "error"
    assert "provider exploded" in inst.result_log[0].text
    # The error is pending notification
    assert manager.check_completed() == "bomb"
    manager.cleanup()
