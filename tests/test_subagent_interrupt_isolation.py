"""Regression tests for parent interrupt isolation from sub-agents."""

import asyncio

import pytest

from harness.cline_agent import ClineAgent
from harness.config import Config


@pytest.mark.asyncio
async def test_background_subagent_disables_interrupt_observation():
    agent = ClineAgent(config=Config(api_url="http://test.invalid", api_key="key"))

    async def fake_loop():
        assert agent._interrupt_enabled is False
        return "done"

    agent._run_loop = fake_loop
    result = await agent.run_message("background task", enable_interrupt=False)

    assert result == "done"
    assert agent._interrupt_enabled is False
