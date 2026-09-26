"""Regression tests for the focused sub-agent completion-notification loop.

When the user is focused on sub-agent X and X completes, the REPL injects
"[SYSTEM: Sub-agent 'X' has completed...]" as input. Routing that message back
into X made X run a new turn, complete again, reset completion_notified, and
re-notify — an infinite loop of report boxes and ☎ lines. The loop is broken
in two places:

1. main.py routes harness-injected input to the parent, never the focused
   sub-agent.
2. SubAgentManager.run() refuses to feed an agent its own completion
   notification as a new turn (defense in depth).

Also covers: TeeWriter line-buffered passthrough (prompt jumping / extra
newlines), kill_process_tree double-reap (rc 255 "exit status already read"),
and shell-tool interrupt gating for background sub-agents.
"""

import asyncio
import inspect
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.config import Config
from harness.sub_agent_manager import SubAgentManager, TeeWriter


def _make_manager(tmp_path):
    config = Config(api_url="http://test.invalid", api_key="k")
    return SubAgentManager(
        config=config,
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / f"{name}.json",
    )


def _completed_instance(name="audit", final="VERDICT: done"):
    """A fake completed SubAgentInstance whose agent.run_message records calls."""
    calls = []

    class FakeAgent:
        async def run_message(self, text, enable_interrupt=True):
            calls.append(text)
            return "new turn output"

    inst = SimpleNamespace(
        name=name,
        agent=FakeAgent(),
        task=None,
        status="completed",
        output="transcript...",
        final_result=final,
        tee=None,
        session_path=None,
        completion_notified=False,
        last_error=None,
    )
    return inst, calls


# ── 1. Manager refuses to feed an agent its own completion notification ────

@pytest.mark.asyncio
async def test_run_refuses_own_completion_notification(tmp_path):
    manager = _make_manager(tmp_path)
    inst, calls = _completed_instance("audit")
    manager._agents["audit"] = inst

    notif = (
        "[SYSTEM: Sub-agent 'audit' has completed its task. "
        "Use get_agent_output(name='audit') to retrieve its full output.]"
    )
    result = await manager.run("audit", notif)

    # Returns the cached final result; NO new agent turn is started.
    assert result == "VERDICT: done"
    assert calls == []
    manager.cleanup()


@pytest.mark.asyncio
async def test_run_allows_normal_input_after_completion(tmp_path):
    manager = _make_manager(tmp_path)
    inst, calls = _completed_instance("audit")
    manager._agents["audit"] = inst

    result = await manager.run("audit", "please double-check the results")
    assert calls == ["please double-check the results"]
    assert result == "new turn output"
    manager.cleanup()


@pytest.mark.asyncio
async def test_run_allows_other_agents_notification(tmp_path):
    """A notification about a DIFFERENT agent is not intercepted... but it
    also must not re-run a completed agent: it waits for the current task,
    then starts a new turn. The REPL-level fix ensures such input never
    reaches a focused agent; the manager-level guard only covers the
    self-notification case."""
    manager = _make_manager(tmp_path)
    inst, calls = _completed_instance("audit")
    manager._agents["audit"] = inst

    notif = "[SYSTEM: Sub-agent 'other' has completed its task.]"
    result = await manager.run("audit", notif)
    # Routed through as a normal turn (REPL fix prevents this in practice).
    assert calls == [notif]
    manager.cleanup()


# ── 2. main.py routes system-injected input to the parent ──────────────────

def test_main_routes_system_input_away_from_focused_agent():
    """Static check: the focused-agent routing must skip harness-injected
    input, and every injection site must set the _system_input flag."""
    import harness.main as hm

    src = inspect.getsource(hm)
    assert "if focused_agent and not _system_input:" in src, (
        "focused-agent routing must not receive harness-injected input"
    )

    # Every completion/cron injection site must flag the input as system.
    assert src.count("_system_input = True") >= 6, (
        "expected at least 6 system-input flag assignments "
        f"(found {src.count('_system_input = True')})"
    )
    # The flag must be reset each REPL iteration.
    assert "_system_input = False" in src


# ── 3. TeeWriter line-buffered passthrough ─────────────────────────────────

class _RecordingStdout(io.StringIO):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, text):
        self.writes.append(text)
        return super().write(text)


def test_tewriter_line_buffers_partial_chunks():
    real = _RecordingStdout()
    tee = TeeWriter(real, dynamic=False)
    tee.active = True

    # Simulate rich emitting a styled line in many small chunks
    # (per ANSI span), then the line's newline.
    tee.write("\x1b[2m")
    tee.write("word ")
    tee.write("\x1b[0m")
    tee.write("more")
    assert real.writes == [], "partial line must be held back, not written"

    tee.write("\n")
    assert len(real.writes) == 1, "complete line must flush as ONE write"
    assert real.writes[0] == "\x1b[2mword \x1b[0mmore\n"
    assert "ANSI sequence must stay intact within one write"

    # Next partial line is held back again
    tee.write("next")
    assert len(real.writes) == 1
    tee.flush()
    assert real.writes[-1] == "next"


def test_tewriter_flushes_partial_line_on_flush():
    real = _RecordingStdout()
    tee = TeeWriter(real, dynamic=False)
    tee.active = True
    tee.write("partial without newline")
    assert real.writes == []
    tee.flush()
    assert real.getvalue() == "partial without newline"


def test_tewriter_inactive_writes_only_buffer():
    real = _RecordingStdout()
    tee = TeeWriter(real, dynamic=False)
    tee.active = False
    tee.write("buffered only\n")
    assert real.writes == []
    assert tee.getvalue() == "buffered only\n"


def test_tewriter_multiple_lines_one_write():
    real = _RecordingStdout()
    tee = TeeWriter(real, dynamic=False)
    tee.active = True
    tee.write("line one\nline two\n")
    assert len(real.writes) == 1
    assert real.writes[0] == "line one\nline two\n"


# ── 4. kill_process_tree must not reap asyncio-owned children ───────────────

def test_kill_process_tree_reap_parent_false(monkeypatch):
    import harness.tools._base as base

    killed = []
    waited_targets = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=True):
            return [FakeProc(200)]

        def kill(self):
            killed.append(self.pid)

    def fake_process(pid):
        return FakeProc(pid)

    def fake_wait_procs(targets, timeout=None):
        waited_targets.append([t.pid for t in targets])
        return (targets, [])

    monkeypatch.setattr(base.psutil, "Process", fake_process)
    monkeypatch.setattr(base.psutil, "wait_procs", fake_wait_procs)

    # Default: parent IS waited (sync subprocess.Popen case)
    base.kill_process_tree(100)
    assert waited_targets[-1] == [200, 100]

    # asyncio-owned: parent must NOT be reaped by psutil
    base.kill_process_tree(100, reap_parent=False)
    assert waited_targets[-1] == [200]

    # The parent itself was still killed in both cases
    assert killed.count(100) == 2


# ── 5. Shell tool interrupt gating for background sub-agents ────────────────

def test_owner_interrupts_respects_disabled_agent():
    from harness.tools.shell import _owner_interrupts

    class OwnerEnabled:
        def interrupts_enabled(self):
            return True

    class OwnerDisabled:
        def interrupts_enabled(self):
            return False

    assert _owner_interrupts(SimpleNamespace(owner_agent=OwnerEnabled())) is True
    assert _owner_interrupts(SimpleNamespace(owner_agent=OwnerDisabled())) is False
    # No owner (legacy/embedded use) → interrupts observed (parent default)
    assert _owner_interrupts(SimpleNamespace()) is True


def test_cline_agent_exposes_interrupts_enabled():
    from harness.cline_agent import ClineAgent

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    assert agent.interrupts_enabled() is True  # default at construction
    agent._interrupt_enabled = False
    assert agent.interrupts_enabled() is False
    # Tool handlers hold a reference to the owning agent
    assert agent.tool_handlers.owner_agent is agent


def test_toolhandlers_owner_agent_default_none():
    from harness.tools import ToolHandlers

    handlers = ToolHandlers(
        config=None,
        console=None,
        workspace_path=".",
        context=None,
        duplicate_detector=None,
    )
    assert handlers.owner_agent is None
