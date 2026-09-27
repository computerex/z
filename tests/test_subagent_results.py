"""Regression tests for concise completed sub-agent result retrieval.

Contract note (v3 UX overhaul): result retrieval belongs to get_agent_output
(consume-on-fetch result log). send_agent_input is now purely a multi-turn
conversation primitive — it starts a new turn on an idle agent and queues on
a busy one; the old "completed agent → return cached result without a new
turn" fast-path was removed because get_agent_output owns that job.
"""

from types import SimpleNamespace

import pytest

from harness.tools.subagent import get_agent_output, send_agent_input


class _Manager:
    def __init__(self, instance):
        self.instance = instance
        self.run_calls = []

    def get(self, name):
        return self.instance if name == "audit" else None

    async def run(self, name, text, wait=True):
        self.run_calls.append((name, text, wait))
        return "new turn output"

    def fetch_new_results(self, name):
        return []


class _FakeConsole:
    def print(self, *a, **k):
        pass


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
async def test_send_agent_input_starts_new_turn_on_completed_agent():
    """send_agent_input on an idle (completed) agent starts a new turn and
    returns its response — multi-turn conversation with sub-agents works."""
    instance = SimpleNamespace(
        status="completed",
        final_result="Audit finished successfully.",
        output="verbose transcript",
        task=SimpleNamespace(done=lambda: True),
    )
    manager = _Manager(instance)
    handlers = SimpleNamespace(sub_agent_manager=manager, console=_FakeConsole())

    result = await send_agent_input(handlers, {"name": "audit", "input": "summarize"})

    assert result == "new turn output"
    assert manager.run_calls == [("audit", "summarize", False)]
