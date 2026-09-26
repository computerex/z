"""Tests that sub-agent execution runs on an independent background loop."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from harness.config import Config
from harness.sub_agent_manager import SubAgentManager


def _make_manager(tmp_path):
    config = Config(api_url="http://test.invalid", api_key="k")
    manager = SubAgentManager(
        config=config,
        console=None,
        workspace=str(tmp_path),
        get_session_path_fn=lambda ws, name: Path(ws) / f"{name}.json",
    )
    return manager


def test_schedule_bg_runs_coroutine_without_main_loop(tmp_path):
    manager = _make_manager(tmp_path)

    async def probe():
        return asyncio.get_running_loop()

    # No running main-loop here: schedule purely on the background thread.
    bg_loop = manager._ensure_bg_loop()
    fut = manager._schedule_bg(probe())
    assert fut.result(timeout=5) is bg_loop
    manager.cleanup()


def test_subagent_progresses_while_main_loop_is_not_pumped(tmp_path):
    """Sub-agent must advance even when the main thread never pumps a loop."""
    manager = _make_manager(tmp_path)

    # Invalid API URL: the agent will fail fast — enough to prove it executed
    # on the background loop without any main-loop pumping.
    manager.create("independent-agent", "do something")
    inst = manager.get("independent-agent")
    assert inst.task is not None

    deadline = time.time() + 30
    while time.time() < deadline:
        if inst.status in ("completed", "error"):
            break
        time.sleep(0.1)

    assert inst.status in ("completed", "error")
    assert inst.task.done()
    manager.cleanup()
