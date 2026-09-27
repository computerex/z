"""Regression tests: sub-agent completion notifications must fire while the
REPL sits idle at the prompt.

The REPL previously only checked for completed sub-agents at fixed points:
at the top of the loop (before the prompt appears) and immediately after the
user pressed Enter. While the prompt was up, the REPL was parked inside
prompt_async() — a sub-agent that finished during that window stayed silent
until the user typed something. The prompt is now raced against a completion
watcher: when a sub-agent completes, the prompt app exits gracefully (typed
text preserved as the next prompt's default) and the notification is
delivered immediately.
"""

import asyncio
import inspect
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


def _completed_instance(name="audit"):
    return SimpleNamespace(
        name=name,
        agent=None,
        task=None,
        status="completed",
        output="transcript",
        final_result="VERDICT: done",
        tee=None,
        session_path=None,
        completion_notified=False,
        last_error=None,
    )


def test_peek_completed_is_non_consuming(tmp_path):
    """peek_completed() must detect a pending notification WITHOUT claiming
    it — the prompt-side watcher peeks, the consuming check_completed()
    after the prompt still owns injecting it."""
    manager = _make_manager(tmp_path)
    manager._agents["audit"] = _completed_instance("audit")

    assert manager.peek_completed() == "audit"
    assert manager.peek_completed() == "audit"  # not consumed
    assert manager.peek_completed() == "audit"  # still not consumed

    # check_completed() still claims it exactly once
    assert manager.check_completed() == "audit"
    assert manager.peek_completed() is None
    assert manager.check_completed() is None
    manager.cleanup()


def test_peek_completed_ignores_already_notified(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _completed_instance("audit")
    inst.completion_notified = True
    manager._agents["audit"] = inst
    assert manager.peek_completed() is None
    assert manager.check_completed() is None
    manager.cleanup()


def test_peek_completed_ignores_running_agents(tmp_path):
    manager = _make_manager(tmp_path)
    inst = _completed_instance("audit")
    inst.status = "running"
    manager._agents["audit"] = inst
    assert manager.peek_completed() is None
    manager.cleanup()


def test_completion_watch_wakes_on_pending_notification():
    """The prompt-side watcher coroutine must return as soon as
    peek_completed() reports a pending notification (simulated manager)."""

    class FakeManager:
        def __init__(self):
            self.ready = asyncio.Event()

        def peek_completed(self):
            if self.ready.is_set():
                return "audit"
            return None

    async def _completion_watch(manager, interval=0.05):
        """Same shape as the one defined in main.py's REPL."""
        while True:
            if manager.peek_completed():
                return True
            await asyncio.sleep(interval)

    async def run():
        mgr = FakeManager()
        task = asyncio.ensure_future(_completion_watch(mgr))
        await asyncio.sleep(0.15)  # watcher polls, finds nothing
        assert not task.done()
        mgr.ready.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        return result

    assert asyncio.run(run()) is True


def test_prompt_races_completion_watcher():
    """Static check: the REPL prompt must be raced against a completion
    watcher so idle-prompt completions wake the parent immediately."""
    import harness.main as hm

    src = inspect.getsource(hm)
    # The watcher polls peek_completed() (non-consuming)
    assert "_completion_watch" in src
    assert "sub_agent_manager.peek_completed()" in src
    # The race: prompt vs watcher, first to finish wins
    assert "return_when=asyncio.FIRST_COMPLETED" in src
    # On watcher win: prompt exits gracefully, typed text preserved
    assert "_app.exit(result=\"\")" in src
    assert "_preserved_typed[0] = _typed" in src
    # Preserved text is restored as the next prompt's default
    assert "default=_preserved_typed[0]" in src
