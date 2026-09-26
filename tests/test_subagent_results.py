"""Regression tests for concise completed sub-agent result retrieval."""

from types import SimpleNamespace

import pytest

from harness.tools.subagent import get_agent_output, send_agent_input


class _Manager:
    def __init__(self, instance):
        self.instance = instance

    def get(self, name):
        return self.instance if name == "audit" else None


@pytest.mark.asyncio
async def test_get_agent_output_prefers_final_result_over_terminal_transcript():
    instance = SimpleNamespace(
        status="completed",
        final_result="Verified migration; no stale endpoints remain.",
        output="\x1b[2mvery long ANSI-decorated transcript\x1b[0m",
        tee=None,
        task=None,
    )
    handlers = SimpleNamespace(sub_agent_manager=_Manager(instance))

    result = await get_agent_output(handlers, {"name": "audit"})

    assert result == "Verified migration; no stale endpoints remain."


@pytest.mark.asyncio
async def test_send_agent_input_retrieves_final_result_without_new_turn():
    instance = SimpleNamespace(
        status="completed",
        final_result="Audit finished successfully.",
        output="verbose transcript",
        task=SimpleNamespace(done=lambda: True),
    )
    handlers = SimpleNamespace(sub_agent_manager=_Manager(instance), console=None)

    result = await send_agent_input(handlers, {"name": "audit", "input": "summarize"})

    assert result == "Audit finished successfully."
