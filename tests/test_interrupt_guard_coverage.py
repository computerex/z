"""Regression tests: no is_interrupted() check may fire for interrupt-disabled agents.

Sub-agents run with enable_interrupt=False and must never observe the parent's
global keyboard interrupt state. Two unguarded checks previously let a stale
escape flag make sub-agents return "[Interrupted - session preserved...]" as
their final result, so the parent retrieved a bogus verdict.
"""

import asyncio
import inspect
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import harness.cline_agent as ca


def test_all_interrupt_checks_are_guarded_by_agent_flag():
    """Every is_interrupted() call site must be gated by self._interrupt_enabled."""
    src = inspect.getsource(ca)
    lines = src.splitlines()
    unguarded = []
    for i, line in enumerate(lines):
        if "is_interrupted()" in line:
            if "_interrupt_enabled" in line:
                continue
            # Allow the definition/import lines and the flag-parameter usage.
            if "check_interrupt" in line or "import" in line or "def " in line:
                continue
            unguarded.append((i + 1, line.strip()[:100]))
    assert not unguarded, f"Unguarded interrupt checks: {unguarded}"


def test_interrupt_disabled_agent_ignores_global_interrupt(monkeypatch):
    """A disabled agent's run_message must not bail when the global flag is set."""
    from harness.config import Config
    from harness.interrupt import get_interrupt_state

    agent = ca.ClineAgent(config=Config(api_url="http://test.invalid", api_key="k"))

    async def fake_loop():
        # Simulate the pre-tool-execution interrupt check with the flag set.
        get_interrupt_state().trigger("escape")
        for _ in range(10):
            if agent._interrupt_enabled and ca.is_interrupted():
                return "[Interrupted - session preserved]"
            return "completed normally"

    agent._run_loop = fake_loop
    result = asyncio.run(agent.run_message("task", enable_interrupt=False))
    assert result == "completed normally"
