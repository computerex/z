"""Regression tests for sub-agent persistence: registry save/restore,
lazy hydration, and dangling tool-call stitching.

Previously sub-agent sessions were written to .sessions/_sub_<name>.json but
NEVER restored — a Ctrl+C-restart lost all sub-agents even though the data
survived on disk. Now a registry (.sessions/_subagents.json) tracks agents
across runs; restore adopts metadata only (lazy hydration — ClineAgent
construction is expensive) and repairs history that was cut mid-turn.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.config import Config
from harness.sub_agent_manager import SubAgentManager, SubAgentInstance


def _make_manager(tmp_path):
    return SubAgentManager(
        config=Config(api_url="http://test.invalid", api_key="k"),
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / ".sessions" / f"{name}.json",
    )


# ── Registry roundtrip ───────────────────────────────────────────────────────

def test_registry_written_on_create(tmp_path):
    manager = _make_manager(tmp_path)

    class FakeAgent:
        async def run_message(self, text, enable_interrupt=True):
            return "done"

    # Build a minimal instance through the manager's own path
    inst = SubAgentInstance(
        name="audit",
        agent=FakeAgent(),
        session_path=Path(tmp_path) / ".sessions" / "_sub_audit.json",
    )
    manager._agents["audit"] = inst
    manager._write_registry()

    data = json.loads((Path(tmp_path) / ".sessions" / "_subagents.json").read_text())
    assert data["agents"][0]["name"] == "audit"
    assert data["agents"][0]["safe_name"] == "audit"
    manager.cleanup()


def test_registry_roundtrip_completed(tmp_path):
    manager = _make_manager(tmp_path)
    inst = SubAgentInstance(
        name="audit",
        agent=None,
        status="completed",
        final_result="the verdict",
        session_path=Path(tmp_path) / ".sessions" / "_sub_audit.json",
    )
    manager._agents["audit"] = inst
    manager._write_registry()

    # New manager (fresh run) restores from the registry
    manager2 = _make_manager(tmp_path)
    n = manager2.restore()
    assert n == 1
    restored = manager2.get("audit")
    assert restored is not None
    assert restored.status == "completed"
    assert restored.final_result == "the verdict"
    assert restored.hydrated is False  # lazy
    assert restored.agent is None
    assert restored.restored is True
    assert restored.completion_notified is True
    assert manager2.peek_completed() is None
    manager.cleanup()
    manager2.cleanup()


@pytest.mark.asyncio
async def test_restore_notice_waits_for_user_turn(monkeypatch):
    from harness.cline_agent import ClineAgent

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent.queue_system_message("[SYSTEM: restored agents available]")
    assert not agent.has_queued_cron_prompts()
    assert not agent.messages  # startup notice alone does not start a turn

    monkeypatch.setattr(agent, "_ensure_cron_scheduler", lambda: None)
    async def capture_turn():
        return "done"
    monkeypatch.setattr(agent, "_run_loop", capture_turn)
    await agent.run_message("hello", enable_interrupt=False)
    assert [m.content for m in agent.messages[-2:]] == [
        "[SYSTEM: restored agents available]", "hello",
    ]


def test_restore_interrupted_status(tmp_path):
    manager = _make_manager(tmp_path)
    inst = SubAgentInstance(
        name="worker",
        agent=None,
        status="interrupted",
        session_path=Path(tmp_path) / ".sessions" / "_sub_worker.json",
    )
    manager._agents["worker"] = inst
    manager._write_registry()

    manager2 = _make_manager(tmp_path)
    manager2.restore()
    assert manager2.get("worker").status == "interrupted"
    manager.cleanup()
    manager2.cleanup()


def test_restore_skips_missing_session_files_gracefully(tmp_path):
    manager = _make_manager(tmp_path)
    # Registry entry pointing at a session file that vanished
    reg_dir = Path(tmp_path) / ".sessions"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "_subagents.json").write_text(json.dumps({
        "agents": [{
            "name": "ghost",
            "safe_name": "ghost",
            "status": "completed",
            "session_path": str(reg_dir / "_sub_ghost.json"),
            "final_result": "old result",
        }]
    }))
    n = manager.restore()
    assert n == 1
    ghost = manager.get("ghost")
    assert ghost.session_path is None  # vanished file → no path kept
    assert ghost.final_result == "old result"  # still inspectable via registry
    manager.cleanup()


def test_restore_corrupt_registry_skips(tmp_path):
    manager = _make_manager(tmp_path)
    reg_dir = Path(tmp_path) / ".sessions"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "_subagents.json").write_text("{not json")
    assert manager.restore() == 0
    manager.cleanup()


def test_restore_name_collision_live_wins(tmp_path):
    manager = _make_manager(tmp_path)
    # Live agent named "audit" exists
    live = SubAgentInstance(name="audit", agent=None, session_path=None)
    manager._agents["audit"] = live
    # Registry also has an "audit" from a previous run
    reg_dir = Path(tmp_path) / ".sessions"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "_subagents.json").write_text(json.dumps({
        "agents": [
            {"name": "audit", "safe_name": "audit", "status": "completed",
             "session_path": None, "final_result": "stale"},
            {"name": "other", "safe_name": "other", "status": "completed",
             "session_path": None, "final_result": "kept"},
        ]
    }))
    n = manager.restore()
    assert n == 1  # only "other" restored
    assert manager.get("audit") is live  # live instance wins
    manager.cleanup()


def test_cleanup_classifies_running_as_interrupted(tmp_path):
    manager = _make_manager(tmp_path)

    class NeverDoneTask:
        def done(self):
            return False

        def cancel(self):
            pass

    inst = SubAgentInstance(name="busy", agent=None, session_path=None)
    inst.task = NeverDoneTask()
    inst.status = "running"
    manager._agents["busy"] = inst
    manager.cleanup()  # writes registry with interrupted status

    data = json.loads((Path(tmp_path) / ".sessions" / "_subagents.json").read_text())
    entry = next(a for a in data["agents"] if a["name"] == "busy")
    assert entry["status"] == "interrupted"


# ── Lazy hydration ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hydrate_on_input_starts_turn_with_history(tmp_path):
    manager = _make_manager(tmp_path)
    # A restored instance with a saved session (write a minimal session file)
    session_dir = Path(tmp_path) / ".sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "_sub_audit.json"
    session_path.write_text(json.dumps({"messages": []}))  # load_session will inject system prompt

    inst = SubAgentInstance(
        name="audit",
        agent=None,
        status="interrupted",
        session_path=session_path,
        hydrated=False,
        restored=True,
    )
    manager._agents["audit"] = inst

    # First interaction hydrates: agent constructed lazily, then the turn runs
    result = await manager.run("audit", "continue where you left off")
    assert inst.hydrated is True
    assert inst.agent is not None
    assert result  # the FakeAgent-free path: run_message against a fresh agent with test config

    # The agent's history was loaded and system prompt ensured
    assert inst.agent._initialized is True
    manager.cleanup()


def test_hydrate_failure_returns_false_no_crash(tmp_path):
    manager = _make_manager(tmp_path)
    # A restored instance whose session path points at a corrupt file must
    # still hydrate (fresh agent) without crashing; hydration "failure" is
    # only about agent construction. Simulate a construction failure with a
    # config that explodes.
    class ExplodingConfig(Config):
        @property
        def api_key(self):
            raise RuntimeError("boom")

        @api_key.setter
        def api_key(self, v):
            pass

    manager._config = ExplodingConfig(api_url="http://test.invalid", api_key="k")
    inst = SubAgentInstance(name="broken", agent=None, session_path=None, hydrated=False)
    manager._agents["broken"] = inst
    ok = manager._hydrate(inst)
    assert ok is False  # graceful failure, no exception
    manager.cleanup()


# ── Dangling tool-call stitching ─────────────────────────────────────────────

def test_stitch_dangling_tool_calls_injects_synthetic_results():
    from harness.cline_agent import ClineAgent
    from harness.streaming_client import StreamingMessage

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent.messages = [
        StreamingMessage(role="system", content="sys"),
        StreamingMessage(role="user", content="do the thing"),
        # Assistant emitted a tool call but was killed before the result
        StreamingMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"}}],
        ),
    ]

    SubAgentManager._stitch_dangling_tool_calls(agent)

    # A synthetic tool result for the dangling call must exist
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_1"
    assert tool_msgs[0].name == "read_file"
    assert "not executed" in tool_msgs[0].content


def test_stitch_drops_orphan_tool_results():
    from harness.cline_agent import ClineAgent
    from harness.streaming_client import StreamingMessage

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent.messages = [
        StreamingMessage(role="system", content="sys"),
        StreamingMessage(role="user", content="hi"),
        # Orphan tool result — its assistant message was evicted
        StreamingMessage(role="tool", content="stale", tool_call_id="gone_1",
                         name="search_files"),
    ]

    SubAgentManager._stitch_dangling_tool_calls(agent)

    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert tool_msgs == []  # orphan dropped


def test_stitch_noop_on_healthy_history():
    from harness.cline_agent import ClineAgent
    from harness.streaming_client import StreamingMessage

    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))
    agent.messages = [
        StreamingMessage(role="system", content="sys"),
        StreamingMessage(role="user", content="hi"),
        StreamingMessage(
            role="assistant", content="",
            tool_calls=[{"id": "call_1", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}],
        ),
        StreamingMessage(role="tool", content="file contents", tool_call_id="call_1",
                         name="read_file"),
    ]
    before = list(agent.messages)

    SubAgentManager._stitch_dangling_tool_calls(agent)

    assert [m.role for m in agent.messages] == [m.role for m in before]
    assert len(agent.messages) == len(before)


# ── Transcript tail ───────────────────────────────────────────────────────────

def test_transcript_tail_written_on_completion(tmp_path):
    manager = _make_manager(tmp_path)
    session_dir = Path(tmp_path) / ".sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "_sub_audit.json"

    from harness.sub_agent_manager import TeeWriter

    tee = TeeWriter(None)
    tee.write("agent output line 1\nline 2\n")
    inst = SubAgentInstance(
        name="audit",
        agent=None,
        tee=tee,
        session_path=session_path,
    )
    manager._agents["audit"] = inst
    manager._save_transcript_tail(inst)

    tail = session_path.with_suffix(".log")
    assert tail.exists()
    assert "agent output line 1" in tail.read_text(encoding="utf-8")
    manager.cleanup()


# ── purge ────────────────────────────────────────────────────────────────────

def test_purge_removes_only_restored(tmp_path):
    manager = _make_manager(tmp_path)
    live = SubAgentInstance(name="live", agent=None, session_path=None)
    restored = SubAgentInstance(name="old", agent=None, session_path=None,
                                restored=True)
    manager._agents["live"] = live
    manager._agents["old"] = restored

    n = manager.purge_restored()
    assert n == 1
    assert manager.get("live") is live
    assert manager.get("old") is None
    manager.cleanup()
