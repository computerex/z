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


def test_schedule_bg_prefers_running_main_loop(tmp_path):
    """With a running loop, sub-agent tasks must be scheduled on it.

    asyncio subprocess child watchers only work on the main thread's loop
    on Unix, so main-loop scheduling is required for shell tools to work.
    """
    manager = _make_manager(tmp_path)

    async def probe():
        return asyncio.get_running_loop()

    async def run():
        task = manager._schedule_bg(probe())
        assert isinstance(task, asyncio.Task)
        return await manager._await_task(task)

    loop_used = asyncio.run(run())
    assert loop_used is not None
    manager.cleanup()


def test_schedule_bg_falls_back_to_dedicated_loop_without_running_loop(tmp_path):
    """No running loop (sync context) → schedule on the dedicated loop thread."""
    manager = _make_manager(tmp_path)

    async def probe():
        return asyncio.get_running_loop()

    fut = manager._schedule_bg(probe())  # no running loop here
    bg_loop = manager._ensure_bg_loop()
    assert fut.result(timeout=5) is bg_loop
    manager.cleanup()


def test_subagent_scheduled_and_progresses_on_main_loop(tmp_path):
    manager = _make_manager(tmp_path)

    async def run():
        manager.create("main-loop-agent", "do something")
        inst = manager.get("main-loop-agent")
        assert isinstance(inst.task, asyncio.Task)

        deadline = time.time() + 30
        while time.time() < deadline:
            if inst.status in ("completed", "error"):
                break
            await asyncio.sleep(0.1)
        assert inst.status in ("completed", "error")
        assert inst.task.done()

    asyncio.run(run())
    manager.cleanup()
