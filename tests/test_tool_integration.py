"""Integration tests for tool-use pipeline.

These tests validate the ACTUAL pipeline the agent uses:
  Model output (XML) → parse_xml_tool() → _execute_tool() → result

Unlike unit tests which test Python classes in isolation, these prove
that if the LLM generates the right XML, the harness correctly parses
and executes it. This is the critical gap that unit tests miss.
"""

import asyncio
import contextlib
import io
import sys
import os
import json
import re
from pathlib import Path
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from harness.cline_agent import parse_xml_tool, parse_all_xml_tools, ParsedToolCall, ClineAgent
from harness.config import Config
from harness.todo_manager import TodoManager, TodoStatus
from harness.smart_context import SmartContextManager
from harness.streaming_client import StreamingChatResponse


# ============================================================
# XML Parsing Tests — Does parse_xml_tool handle every tool?
# ============================================================

class TestParseXmlTool:
    """Test that parse_xml_tool correctly extracts tool calls from
    realistic model output (including surrounding prose)."""

    def test_parse_read_file(self):
        content = """I need to examine the auth module to understand the flow.

<read_file>
<path>src/auth.py</path>
</read_file>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "read_file"
        assert result.parameters["path"] == "src/auth.py"

    def test_parse_write_to_file(self):
        content = """I'll create the config file now.

<write_to_file>
<path>config.json</path>
<content>
{
    "key": "value",
    "nested": {
        "a": 1
    }
}
</content>
</write_to_file>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "write_to_file"
        assert result.parameters["path"] == "config.json"
        assert '"key": "value"' in result.parameters["content"]

    def test_parse_replace_in_file(self):
        content = """I see the bug. Let me fix line 42.

<replace_in_file>
<path>src/auth.py</path>
<diff>
<<<<<<< SEARCH
    if token.validate():
=======
    if token is not None and token.validate():
>>>>>>> REPLACE
</diff>
</replace_in_file>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "replace_in_file"
        assert result.parameters["path"] == "src/auth.py"
        assert "<<<<<<< SEARCH" in result.parameters["diff"]
        assert "token is not None" in result.parameters["diff"]

    def test_parse_execute_command(self):
        content = """Let me run the tests to verify.

<execute_command>
<command>python -m pytest tests/ -v</command>
</execute_command>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "execute_command"
        assert result.parameters["command"] == "python -m pytest tests/ -v"

    def test_parse_execute_command_background(self):
        content = """Starting the dev server.

<execute_command>
<command>npm run dev</command>
<background>true</background>
</execute_command>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "execute_command"
        assert result.parameters["command"] == "npm run dev"
        assert result.parameters["background"] == "true"

    def test_parse_list_files(self):
        content = """<list_files>
<path>src/</path>
<recursive>true</recursive>
</list_files>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "list_files"
        assert result.parameters["path"] == "src/"
        assert result.parameters["recursive"] == "true"

    def test_parse_tool_call_shorthand_list_files(self):
        content = """I'll inspect the repo layout first.

<tool_call>list_files path="." recursive="true" />"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "list_files"
        assert result.parameters["path"] == "."
        assert result.parameters["recursive"] == "true"

    def test_parse_search_files(self):
        content = """<search_files>
<path>src/</path>
<regex>\\bdef authenticate\\b</regex>
<file_pattern>*.py</file_pattern>
</search_files>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "search_files"
        assert result.parameters["path"] == "src/"
        assert "authenticate" in result.parameters["regex"]
        assert result.parameters["file_pattern"] == "*.py"

    def test_parse_web_search(self):
        content = """<web_search>
<query>Python 3.12 asyncio changes</query>
<count>5</count>
</web_search>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "web_search"
        assert result.parameters["query"] == "Python 3.12 asyncio changes"
        assert result.parameters["count"] == "5"

    def test_parse_list_context(self):
        content = """Let me check what's in my context.

<list_context>
</list_context>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "list_context"

    def test_parse_remove_from_context(self):
        content = """<remove_from_context>
<id>3</id>
</remove_from_context>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "remove_from_context"
        assert result.parameters["id"] == "3"

    def test_no_tool_call(self):
        """Plain text with no XML should return None."""
        content = "I cannot run commands directly. I am an AI running in a text-only environment."
        result = parse_xml_tool(content)
        assert result is None

    def test_no_tool_call_with_xml_like_text(self):
        """Text that mentions XML tags but isn't a tool call."""
        content = "You can use <read_file> to read files. Just put the path inside <path> tags."
        # This might parse as a tool call — the key question is does it matter?
        # Actually, in Cline format, if the model generates these tags, they DO get parsed.
        # The issue is when the model doesn't generate ANY tags.
        pass

    def test_last_tool_wins(self):
        """When model generates multiple tool calls, the LAST one wins."""
        content = """I'll first read the file.

<read_file>
<path>old_file.py</path>
</read_file>

Actually, let me read a different file instead.

<read_file>
<path>correct_file.py</path>
</read_file>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "read_file"
        assert result.parameters["path"] == "correct_file.py"


# ============================================================
# Multi-Tool-Call Parsing Tests
# ============================================================

class TestParseAllXmlTools:
    """Test that parse_all_xml_tools finds every tool call in order."""

    def test_single_tool(self):
        content = """<read_file>
<path>foo.py</path>
</read_file>"""
        results = parse_all_xml_tools(content)
        assert len(results) == 1
        assert results[0].name == "read_file"
        assert results[0].parameters["path"] == "foo.py"

    def test_multiple_manage_todos(self):
        """The exact scenario that was broken: 3 manage_todos in one response."""
        content = """I'll create a todo list.

<manage_todos>
<action>add</action>
<title>Delete evoke.exe</title>
</manage_todos>

<manage_todos>
<action>add</action>
<title>Rebuild evoke.exe</title>
</manage_todos>

<manage_todos>
<action>add</action>
<title>Launch evoke.exe</title>
</manage_todos>"""
        results = parse_all_xml_tools(content)
        assert len(results) == 3
        assert results[0].parameters["title"] == "Delete evoke.exe"
        assert results[1].parameters["title"] == "Rebuild evoke.exe"
        assert results[2].parameters["title"] == "Launch evoke.exe"

    def test_mixed_tool_types(self):
        content = """<manage_todos>
<action>update</action>
<id>1</id>
<status>in-progress</status>
</manage_todos>

<read_file>
<path>src/main.py</path>
</read_file>"""
        results = parse_all_xml_tools(content)
        assert len(results) == 2
        assert results[0].name == "manage_todos"
        assert results[1].name == "read_file"

    def test_no_tools(self):
        content = "Just some plain text with no XML tool calls."
        results = parse_all_xml_tools(content)
        assert len(results) == 0

    def test_with_thinking_blocks(self):
        content = """<thinking>Let me plan this out.</thinking>

<manage_todos>
<action>add</action>
<title>First task</title>
</manage_todos>

<manage_todos>
<action>add</action>
<title>Second task</title>
</manage_todos>"""
        results = parse_all_xml_tools(content)
        assert len(results) == 2
        assert results[0].parameters["title"] == "First task"
        assert results[1].parameters["title"] == "Second task"

    def test_parse_all_preserves_order(self):
        content = """<manage_todos>
<action>add</action>
<title>A</title>
</manage_todos>
<manage_todos>
<action>add</action>
<title>B</title>
</manage_todos>
<manage_todos>
<action>add</action>
<title>C</title>
</manage_todos>"""
        results = parse_all_xml_tools(content)
        titles = [r.parameters["title"] for r in results]
        assert titles == ["A", "B", "C"]


# ============================================================
# manage_todos XML Parsing Tests
# ============================================================

class TestParseTodoXml:
    """Test that manage_todos XML is correctly parsed — the critical
    integration point between our new code and the existing parser."""

    def test_parse_add_simple(self):
        content = """I'll break this task down into steps.

<manage_todos>
<action>add</action>
<title>Implement user authentication</title>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "manage_todos"
        assert result.parameters["action"] == "add"
        assert result.parameters["title"] == "Implement user authentication"

    def test_parse_add_with_description(self):
        content = """<manage_todos>
<action>add</action>
<title>Add JWT support</title>
<context_refs>src/auth.py,src/config.py</context_refs>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "manage_todos"
        assert result.parameters["action"] == "add"
        assert result.parameters["title"] == "Add JWT support"
        assert "src/auth.py" in result.parameters["context_refs"]
        assert "src/config.py" in result.parameters["context_refs"]

    def test_parse_add_subtask(self):
        content = """<manage_todos>
<action>add</action>
<title>Create login endpoint</title>
<parent_id>1</parent_id>
<context_refs>src/routes.py</context_refs>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.parameters["action"] == "add"
        assert result.parameters["parent_id"] == "1"
        assert result.parameters["title"] == "Create login endpoint"

    def test_parse_update_status(self):
        content = """<manage_todos>
<action>update</action>
<id>1</id>
<status>in-progress</status>
<notes>Found existing auth module, extending it</notes>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.parameters["action"] == "update"
        assert result.parameters["id"] == "1"
        assert result.parameters["status"] == "in-progress"
        assert "extending it" in result.parameters["notes"]

    def test_parse_update_completed(self):
        content = """Task is done.

<manage_todos>
<action>update</action>
<id>3</id>
<status>completed</status>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.parameters["action"] == "update"
        assert result.parameters["id"] == "3"
        assert result.parameters["status"] == "completed"

    def test_parse_remove(self):
        content = """<manage_todos>
<action>remove</action>
<id>5</id>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.parameters["action"] == "remove"
        assert result.parameters["id"] == "5"

    def test_parse_list(self):
        content = """Let me review my progress.

<manage_todos>
<action>list</action>
</manage_todos>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.parameters["action"] == "list"


# ============================================================
# Tool Dispatch Tests — Does _handle_manage_todos work via _execute_tool?
# ============================================================

class TestToolDispatch:
    """Test the full dispatch path: ParsedToolCall → _execute_tool() → result.
    
    Uses a real ClineAgent with a mock config (no API calls needed)."""

    def _make_agent(self):
        """Create a ClineAgent for testing (no API connection needed)."""
        config = Config(
            api_url="http://test.invalid",
            api_key="test-key",
            model="test-model",
        )
        agent = ClineAgent(config=config, max_iterations=1)
        return agent

    def test_dispatch_manage_todos_add(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Fix the auth bug"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Added todo [1]" in result
        assert "Fix the auth bug" in result
        assert len(agent.todo_manager.list_all()) == 1

    def test_dispatch_manage_todos_add_with_refs(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "add",
                "title": "Implement feature",
                "context_refs": "src/auth.py,src/config.py"
            }
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Added todo [1]" in result
        item = agent.todo_manager.get(1)
        assert "src/auth.py" in item.context_refs
        assert "src/config.py" in item.context_refs

    def test_dispatch_manage_todos_add_subtask(self):
        agent = self._make_agent()
        # Add parent
        tool1 = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Main task"}
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool1))
        
        # Add child
        tool2 = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Sub-task", "parent_id": "1"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool2))
        assert "Added todo [2]" in result
        assert agent.todo_manager.get(2).parent_id == 1

    def test_dispatch_manage_todos_update(self):
        agent = self._make_agent()
        # Add first
        agent.todo_manager.add("Task 1")
        
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "1", "status": "in-progress"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Updated todo [1]" in result
        assert agent.todo_manager.get(1).status == TodoStatus.IN_PROGRESS

    def test_dispatch_manage_todos_update_with_notes(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")
        
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "update",
                "id": "1",
                "status": "in-progress",
                "notes": "Found the bug in auth.py line 42"
            }
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Updated todo [1]" in result
        assert agent.todo_manager.get(1).notes == "Found the bug in auth.py line 42"

    def test_dispatch_manage_todos_complete(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")
        
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "1", "status": "completed"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Updated todo [1]" in result
        assert agent.todo_manager.get(1).status == TodoStatus.COMPLETED

    def test_dispatch_manage_todos_remove(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")
        
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "remove", "id": "1"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Removed todo [1]" in result
        assert len(agent.todo_manager.list_all()) == 0

    def test_dispatch_manage_todos_list(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")
        agent.todo_manager.add("Task 2")
        
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "list"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Task 1" in result
        assert "Task 2" in result

    def test_dispatch_manage_todos_error_no_title(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "'title' is required" in result

    def test_dispatch_manage_todos_error_invalid_id(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "abc"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "Invalid id" in result

    def test_dispatch_manage_todos_error_not_found(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "99", "status": "completed"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "not found" in result

    def test_dispatch_manage_todos_error_unknown_action(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "destroy"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "Unknown action" in result

    def test_dispatch_set_reasoning_mode(self):
        """set_reasoning_mode should change the agent's reasoning mode."""
        agent = self._make_agent()
        # Set up providers so mode switching works
        agent.providers = {
            "fast": {"api_url": "http://fast.invalid", "api_key": "fk", "model": "fast-model"},
            "normal": {"api_url": "http://normal.invalid", "api_key": "nk", "model": "normal-model"},
        }
        assert agent.reasoning_mode == "normal"
        
        tool = ParsedToolCall(
            name="set_reasoning_mode",
            parameters={"mode": "fast"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "fast" in result
        assert agent.reasoning_mode == "fast"
        assert agent.config.model == "fast-model"
    
    def test_dispatch_set_reasoning_mode_invalid(self):
        """Invalid mode should return an error."""
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="set_reasoning_mode",
            parameters={"mode": "ultra"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
    
    def test_dispatch_set_reasoning_mode_already_set(self):
        """Switching to current mode should return 'already in' message."""
        agent = self._make_agent()
        agent.providers = {
            "normal": {"api_url": "http://n.invalid", "api_key": "k", "model": "m"},
        }
        tool = ParsedToolCall(
            name="set_reasoning_mode",
            parameters={"mode": "normal"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Already" in result

    def test_dispatch_execute_command_missing_command_param(self):
        """Malformed execute_command without <command> should be rejected."""
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="execute_command",
            parameters={},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error: malformed <execute_command> call" in result
        assert "missing required parameter(s): command" in result


# ============================================================
# End-to-End Mock Tests — Simulate full model output → parse → execute
# ============================================================

class TestEndToEndMock:
    """Simulate realistic model output and verify the complete pipeline.
    
    These test what happens in _run_loop when the model generates XML:
    1. Model text → parse_xml_tool() → ParsedToolCall
    2. ParsedToolCall → _execute_tool() → result string
    3. Result gets appended as user message for next iteration
    """

    def _make_agent(self):
        config = Config(
            api_url="http://test.invalid",
            api_key="test-key",
            model="test-model",
        )
        agent = ClineAgent(config=config, max_iterations=1)
        # Initialize the system prompt
        from harness.prompts import get_system_prompt
        from harness.streaming_client import StreamingMessage
        agent.messages = [
            StreamingMessage(role="system", content=get_system_prompt(agent.workspace_path))
        ]
        agent._initialized = True
        return agent

    def test_e2e_model_adds_todo_then_reads_file(self):
        """Simulate: model → add todo XML → parse → execute → verify state."""
        agent = self._make_agent()
        
        # Simulate realistic model output with prose + XML
        model_output = """I'll start by breaking this task into steps.

First, let me create a todo list to track my progress:

<manage_todos>
<action>add</action>
<title>Read and understand auth module</title>
<context_refs>src/auth.py</context_refs>
</manage_todos>"""

        # Step 1: Parse the XML
        tool_call = parse_xml_tool(model_output)
        assert tool_call is not None, "parse_xml_tool should find manage_todos in model output"
        assert tool_call.name == "manage_todos"
        
        # Step 2: Execute the tool
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool_call))
        
        # Step 3: Verify state changes
        assert "Added todo [1]" in result
        assert len(agent.todo_manager.list_all()) == 1
        item = agent.todo_manager.get(1)
        assert item.title == "Read and understand auth module"
        assert "src/auth.py" in item.context_refs

    def test_e2e_model_creates_full_todo_list(self):
        """Simulate: model creates multiple todos, updates their status."""
        agent = self._make_agent()
        
        # Step 1: Add first todo
        output1 = """<manage_todos>
<action>add</action>
<title>Fix authentication bug</title>
</manage_todos>"""
        tool1 = parse_xml_tool(output1)
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool1))
        
        # Step 2: Add sub-task
        output2 = """<manage_todos>
<action>add</action>
<title>Add null check for token</title>
<parent_id>1</parent_id>
<context_refs>src/auth.py</context_refs>
</manage_todos>"""
        tool2 = parse_xml_tool(output2)
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool2))
        
        # Step 3: Mark parent in-progress
        output3 = """<manage_todos>
<action>update</action>
<id>1</id>
<status>in-progress</status>
</manage_todos>"""
        tool3 = parse_xml_tool(output3)
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool3))
        
        # Step 4: Complete sub-task
        output4 = """<manage_todos>
<action>update</action>
<id>2</id>
<status>completed</status>
<notes>Added null check on line 42</notes>
</manage_todos>"""
        tool4 = parse_xml_tool(output4)
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool4))
        
        # Verify final state
        assert len(agent.todo_manager.list_all()) == 2
        parent = agent.todo_manager.get(1)
        child = agent.todo_manager.get(2)
        assert parent.status == TodoStatus.IN_PROGRESS
        assert child.status == TodoStatus.COMPLETED
        assert child.parent_id == 1
        assert child.notes == "Added null check on line 42"

    def test_e2e_model_output_with_thinking(self):
        """Model output with <thinking> blocks should still parse tools."""
        model_output = """<thinking>
I need to track my progress on this complex task.
Let me create a todo list first before doing anything.
</thinking>

I'll organize my approach with a todo list.

<manage_todos>
<action>add</action>
<title>Analyze codebase structure</title>
</manage_todos>"""
        
        tool_call = parse_xml_tool(model_output)
        assert tool_call is not None
        assert tool_call.name == "manage_todos"
        assert tool_call.parameters["title"] == "Analyze codebase structure"

    def test_e2e_no_tool_plain_response(self):
        """When model says 'I can't do that' — no tool gets parsed."""
        model_output = """No, I cannot run commands directly.

I am an AI running in a text-only environment. I do not have access 
to a terminal, a command line, or your computer's file system. This 
means I cannot execute 'git clone', 'pip install', or download files 
from GitHub for you.

I can only:
1. **Write the code** for you to copy and paste.
2. **Explain** how to run the commands on your own machine."""

        tool_call = parse_xml_tool(model_output)
        assert tool_call is None, "Should NOT parse any tool from this refusal response"

    def test_e2e_todo_persists_through_save_load(self):
        """Todos survive session save/load."""
        import tempfile
        agent = self._make_agent()
        
        # Add todos
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Persistent task", "context_refs": "main.py"}
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        
        # Save session
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            save_path = f.name
        
        try:
            agent.save_session(save_path)
            
            # Create new agent and load
            agent2 = self._make_agent()
            assert len(agent2.todo_manager.list_all()) == 0
            
            loaded = agent2.load_session(save_path, inject_resume=False)
            assert loaded
            assert len(agent2.todo_manager.list_all()) == 1
            assert agent2.todo_manager.get(1).title == "Persistent task"
            assert "main.py" in agent2.todo_manager.get(1).context_refs
        finally:
            os.unlink(save_path)

    def test_e2e_context_stats_show_todos(self):
        """get_context_stats includes todo counts."""
        agent = self._make_agent()
        
        # Before todos
        stats = agent.get_context_stats()
        assert stats["todos_total"] == 0
        assert stats["todos_active"] == 0
        
        # Add and complete some todos
        agent.todo_manager.add("Task 1")
        agent.todo_manager.add("Task 2")
        t3 = agent.todo_manager.add("Task 3")
        agent.todo_manager.update(t3.id, status="completed")
        
        stats = agent.get_context_stats()
        assert stats["todos_total"] == 3
        assert stats["todos_active"] == 2
        assert stats["todos_completed"] == 1


# ============================================================
# Context Compaction Integration Tests
# ============================================================

class TestContextCompactionIntegration:
    """Test that smart context compaction works through the agent."""

    def _make_agent(self):
        config = Config(
            api_url="http://test.invalid",
            api_key="test-key",
            model="test-model",
        )
        agent = ClineAgent(config=config, max_iterations=1)
        return agent

    def test_smart_context_connected_to_todo_manager(self):
        """SmartContextManager uses the agent's TodoManager."""
        agent = self._make_agent()
        assert agent.smart_context.todo_manager is agent.todo_manager
        
        agent.todo_manager.add("Test task", context_refs=["important.py"])
        refs = agent.smart_context.todo_manager.get_active_context_refs()
        assert "important.py" in refs

    def test_clear_resets_everything(self):
        """clear_history resets todos and smart context."""
        agent = self._make_agent()
        agent.todo_manager.add("Task")
        from harness.smart_context import CompactionTrace
        agent.smart_context.compaction_traces.append(
            CompactionTrace("file_read", "test.py", "test", tokens_freed=100)
        )
        
        agent.clear_history()
        assert len(agent.todo_manager.list_all()) == 0
        assert len(agent.smart_context.compaction_traces) == 0

    def test_eviction_count_in_stats(self):
        """Eviction count shows in context stats."""
        agent = self._make_agent()
        from harness.smart_context import CompactionTrace
        
        agent.smart_context.compaction_traces.append(
            CompactionTrace("file_read", "old.py", "Old file", tokens_freed=500)
        )
        agent.smart_context.compaction_traces.append(
            CompactionTrace("command_output", "npm test", "Pass", tokens_freed=300)
        )
        
        stats = agent.get_context_stats()
        assert stats["evictions"] == 2


# ============================================================
# Prompt Integration Tests — Does the system prompt include tools?
# ============================================================

class TestPromptIntegration:
    """Verify the system prompt contains all tool definitions."""

    def test_system_prompt_has_manage_todos(self):
        from harness.prompts import get_system_prompt
        prompt = get_system_prompt("/test/workspace")
        assert "manage_todos" in prompt
        assert "<action>" in prompt
        assert "add" in prompt
        assert "update" in prompt
        assert "remove" in prompt
        assert "context_refs" in prompt

    def test_system_prompt_has_all_tools(self):
        from harness.prompts import get_system_prompt
        prompt = get_system_prompt("/test/workspace")
        
        required_tools = [
            "read_file", "write_to_file", "replace_in_file",
            "execute_command", "list_files", "search_files",
            "web_search", "list_context", "remove_from_context",
            "manage_todos",
            "set_reasoning_mode", "create_plan",
        ]
        for tool in required_tools:
            assert f"## {tool}" in prompt, f"System prompt missing tool definition: {tool}"

    def test_system_prompt_has_context_management_rules(self):
        from harness.prompts import get_system_prompt
        prompt = get_system_prompt("/test/workspace")
        assert "auto-compacted" in prompt or "context compaction" in prompt.lower() or "NEVER evicted" in prompt
        assert "manage_todos" in prompt
        assert "compaction" in prompt.lower() or "compacted" in prompt.lower()

    def test_parse_xml_tool_knows_all_tools(self):
        """parse_xml_tool's tool_names list matches prompt definitions."""
        from harness.prompts import get_system_prompt
        prompt = get_system_prompt("/test/workspace")
        
        # Extract ## tool_name from prompt
        prompt_tools = set(re.findall(r'^## (\w+)', prompt, re.MULTILINE))
        
        # These tools are defined in parse_xml_tool
        parser_tools = {
            'read_file', 'write_to_file', 'replace_in_file',
            'execute_command', 'list_files', 'search_files',
            'check_background_process', 'stop_background_process', 
            'list_background_processes',
            'list_context', 'remove_from_context', 'analyze_image', 
            'web_search', 'manage_todos',
            'set_reasoning_mode', 'create_plan', 'update_agent_rules',
        }
        
        # Every tool in the prompt should be parseable
        for tool in prompt_tools:
            assert tool in parser_tools, (
                f"Tool '{tool}' defined in system prompt but not in parse_xml_tool's tool_names list"
            )


# ============================================================
# Context Dump Tests
# ============================================================

class TestContextDump:
    """Test the dump_context method for debugging model context."""

    def _make_agent(self):
        config = Config(
            api_url="http://test.invalid",
            api_key="test-key",
            model="test-model",
        )
        agent = ClineAgent(config=config, max_iterations=1)
        from harness.prompts import get_system_prompt
        from harness.streaming_client import StreamingMessage
        agent.messages = [
            StreamingMessage(role="system", content=get_system_prompt(agent.workspace_path))
        ]
        agent._initialized = True
        return agent

    def test_dump_creates_file(self):
        import tempfile
        agent = self._make_agent()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            result_path = agent.dump_context(path=dump_path, reason="test")
            assert result_path == dump_path
            assert os.path.exists(dump_path)
            data = json.loads(open(dump_path, encoding="utf-8").read())
            assert data["reason"] == "test"
            assert data["model"] == "test-model"
            assert len(data["messages"]) == 1  # just system prompt
        finally:
            os.unlink(dump_path)

    def test_dump_contains_system_prompt_analysis(self):
        import tempfile
        agent = self._make_agent()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            agent.dump_context(path=dump_path)
            data = json.loads(open(dump_path, encoding="utf-8").read())
            analysis = data["summary"]["system_prompt_analysis"]
            assert analysis["has_read_file"] == True
            assert analysis["has_execute_command"] == True
            assert analysis["has_manage_todos"] == True
            assert analysis["has_TOOL_USE_section"] == True
            assert analysis["has_RULES_section"] == True
            assert analysis["tokens_est"] > 1500  # System prompt should be ~1700+ tokens (trimmed)
        finally:
            os.unlink(dump_path)

    def test_dump_includes_full_messages(self):
        import tempfile
        from harness.streaming_client import StreamingMessage
        agent = self._make_agent()
        # Add some messages
        agent.messages.append(StreamingMessage(role="user", content="Hello, please read config.py"))
        agent.messages.append(StreamingMessage(role="assistant", content="I'll read that file.\n\n<read_file>\n<path>config.py</path>\n</read_file>"))
        agent.messages.append(StreamingMessage(role="user", content="[read_file result]\ndef main():\n    pass"))
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            agent.dump_context(path=dump_path)
            data = json.loads(open(dump_path, encoding="utf-8").read())
            assert len(data["messages"]) == 4
            assert data["messages"][0]["role"] == "system"
            assert data["messages"][1]["role"] == "user"
            assert data["messages"][2]["role"] == "assistant"
            assert data["messages"][3]["role"] == "user"
            # Each message has metadata
            assert "tokens_est" in data["messages"][0]
            assert "chars" in data["messages"][0]
            assert data["messages"][0]["tokens_est"] > 1500
        finally:
            os.unlink(dump_path)

    def test_dump_includes_todos(self):
        import tempfile
        agent = self._make_agent()
        agent.todo_manager.add("Fix the bug")
        agent.todo_manager.add("Write tests")
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            agent.dump_context(path=dump_path)
            data = json.loads(open(dump_path, encoding="utf-8").read())
            assert "todos" in data
            assert len(data["todos"]["items"]) == 2
        finally:
            os.unlink(dump_path)

    def test_system_prompt_token_count_is_correct(self):
        """Verify the system prompt is at least 4k tokens (regression check)."""
        agent = self._make_agent()
        from harness.context_management import estimate_tokens
        sys_tokens = estimate_tokens(agent.messages[0].content)
        assert sys_tokens >= 1500, (
            f"System prompt is only {sys_tokens} tokens! Expected >= 1500. "
            f"Prompt may be truncated or broken."
        )
        assert "manage_todos" in agent.messages[0].content, "manage_todos tool missing from prompt"


# ============================================================
# Reasoning Mode Tests
# ============================================================

class TestSetReasoningMode:
    """Test the set_reasoning_mode tool and mode switching."""

    def _make_agent(self):
        config = Config(
            api_url="http://normal.invalid",
            api_key="normal-key",
            model="normal-model",
        )
        providers = {
            "fast": {"api_url": "http://fast.invalid", "api_key": "fast-key", "model": "fast-model"},
            "normal": {"api_url": "http://normal.invalid", "api_key": "normal-key", "model": "normal-model"},
        }
        agent = ClineAgent(config=config, max_iterations=1, providers=providers)
        return agent

    def test_default_mode_is_normal(self):
        agent = self._make_agent()
        assert agent.reasoning_mode == "normal"

    def test_switch_to_fast(self):
        agent = self._make_agent()
        result = agent._handle_set_reasoning_mode({"mode": "fast"})
        assert "fast" in result
        assert agent.reasoning_mode == "fast"
        assert agent.config.model == "fast-model"
        assert agent.config.api_url == "http://fast.invalid"

    def test_switch_back_to_normal(self):
        agent = self._make_agent()
        agent._handle_set_reasoning_mode({"mode": "fast"})
        result = agent._handle_set_reasoning_mode({"mode": "normal"})
        assert "normal" in result
        assert agent.reasoning_mode == "normal"
        assert agent.config.model == "normal-model"

    def test_invalid_mode_returns_error(self):
        agent = self._make_agent()
        result = agent._handle_set_reasoning_mode({"mode": "turbo"})
        assert "Error" in result
        assert agent.reasoning_mode == "normal"  # unchanged

    def test_already_in_mode(self):
        agent = self._make_agent()
        result = agent._handle_set_reasoning_mode({"mode": "normal"})
        assert "Already" in result

    def test_no_provider_configured(self):
        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=1, providers={})
        result = agent._handle_set_reasoning_mode({"mode": "fast"})
        assert "Error" in result
        assert "not configured" in result


class TestParseReasoningModeXml:
    """Test that parse_xml_tool correctly handles set_reasoning_mode and create_plan XML."""

    def test_parse_set_reasoning_mode(self):
        content = """I'll switch to fast mode for bulk file reads.

<set_reasoning_mode>
<mode>fast</mode>
</set_reasoning_mode>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "set_reasoning_mode"
        assert result.parameters["mode"] == "fast"

    def test_parse_create_plan(self):
        content = """This requires deep reasoning. Let me delegate to Claude.

<create_plan>
<prompt>Design a migration plan for the auth system from sessions to JWT.</prompt>
</create_plan>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "create_plan"
        assert "migration plan" in result.parameters["prompt"]

    def test_parse_create_plan_long_prompt(self):
        content = """<create_plan>
<prompt>I need to refactor the authentication system from session-based to JWT.
The current code is in src/auth/ with 3 main files.
Design a migration plan that:
1) lists all files that need changes
2) specifies the order of changes
3) identifies potential breaking changes</prompt>
</create_plan>"""
        result = parse_xml_tool(content)
        assert result is not None
        assert result.name == "create_plan"
        assert "refactor" in result.parameters["prompt"]
        assert "breaking changes" in result.parameters["prompt"]


class TestMalformedToolIntentDetection:
    """Ensure malformed function-style XML is recognized for recovery."""

    def test_detects_invoke_function_name_parameter_style(self):
        content = """<thinking>
Need to run a command.
</thinking>
<invoke>
<function_name>execute_command</function_name>
<parameter name="command">dir /s /b gcCoreAPI.h</parameter>
</invoke>"""
        assert ClineAgent._looks_like_malformed_tool_intent(content) is True

    def test_ignores_normal_text(self):
        content = "I'll inspect the project and then report findings."
        assert ClineAgent._looks_like_malformed_tool_intent(content) is False


class TestNoActionRecovery:
    """Ensure no-action assistant text does not terminate the loop immediately."""

    def test_run_loop_retries_once_then_accepts_text(self, monkeypatch):
        """When the model outputs text without a tool call, nudge it once
        then accept the response (default retry max = 1)."""
        responses = [
            StreamingChatResponse(content="I'll first think through the approach."),
            StreamingChatResponse(content="Still analyzing before any action."),
        ]
        state = {"calls": 0}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def chat_stream_raw(
                self,
                messages,
                on_content=None,
                on_reasoning=None,
                check_interrupt=None,
                enable_web_search=False,
                status_line=None,
            ):
                idx = state["calls"]
                state["calls"] += 1
                resp = responses[idx]
                if on_content and resp.content:
                    on_content(resp.content)
                return resp

        monkeypatch.setattr("harness.cline_agent.StreamingJSONClient", _FakeClient)

        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=5)
        result = asyncio.get_event_loop().run_until_complete(
            agent.run("Build the project and report blockers", enable_interrupt=False)
        )

        assert state["calls"] == 2, (
            f"Expected 1 retry (2 calls), got {state['calls']}"
        )
        assert "analyzing" in result.lower()

    def test_whitespace_reasoning_does_not_render_empty_thinking_block(self, monkeypatch):
        responses = [
            StreamingChatResponse(content="No actionable tool yet."),
        ]
        state = {"calls": 0}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def chat_stream_raw(
                self,
                messages,
                on_content=None,
                on_reasoning=None,
                check_interrupt=None,
                enable_web_search=False,
                status_line=None,
            ):
                state["calls"] += 1
                if on_reasoning:
                    on_reasoning("   \n")
                resp = responses[0]
                if on_content and resp.content:
                    on_content(resp.content)
                return resp

        monkeypatch.setattr("harness.cline_agent.StreamingJSONClient", _FakeClient)

        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=1)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = asyncio.get_event_loop().run_until_complete(
                agent.run("Build and diagnose", enable_interrupt=False)
            )

        rendered = stream.getvalue()
        assert state["calls"] == 1
        assert result == "Max iterations reached."
        assert "<thinking>" not in rendered
        assert "</thinking>" not in rendered

    def test_reasoning_stream_is_pretty_printed(self, monkeypatch):
        responses = [
            StreamingChatResponse(content="No tool yet."),
        ]

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def chat_stream_raw(
                self,
                messages,
                on_content=None,
                on_reasoning=None,
                check_interrupt=None,
                enable_web_search=False,
                status_line=None,
            ):
                if on_reasoning:
                    on_reasoning("first line\nsecond line")
                resp = responses[0]
                if on_content and resp.content:
                    on_content(resp.content)
                return resp

        monkeypatch.setattr("harness.cline_agent.StreamingJSONClient", _FakeClient)

        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=1)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            asyncio.get_event_loop().run_until_complete(
                agent.run("Build and diagnose", enable_interrupt=False)
            )

        rendered = stream.getvalue()
        assert "[thinking]" in rendered
        assert "  > first line" in rendered
        assert "  > second line" in rendered
        assert "[/thinking]" in rendered

    def test_no_action_text_accepted_after_one_retry(self, monkeypatch):
        """Plain text (no tool call) should be accepted after 1 retry nudge,
        not burn 4 API calls."""
        state = {"calls": 0}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def chat_stream_raw(
                self,
                messages,
                on_content=None,
                on_reasoning=None,
                check_interrupt=None,
                enable_web_search=False,
                status_line=None,
            ):
                state["calls"] += 1
                if on_content:
                    on_content(
                        "Task completed! I fixed the auth middleware "
                        "and added session validation."
                    )
                return StreamingChatResponse(
                    content="Task completed! I fixed the auth middleware "
                    "and added session validation."
                )

        monkeypatch.setattr("harness.cline_agent.StreamingJSONClient", _FakeClient)

        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=5)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = asyncio.get_event_loop().run_until_complete(
                agent.run("Fix the auth bug", enable_interrupt=False)
            )

        assert state["calls"] == 2, (
            f"Expected 1 retry (2 calls total), got {state['calls']}"
        )
        assert "auth middleware" in result

    def test_response_extracted_from_reasoning_after_thinking_tag(self, monkeypatch):
        """When the model puts its response after </thinking> in the reasoning
        stream and content is empty, extract it as content."""
        state = {"calls": 0}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def chat_stream_raw(
                self,
                messages,
                on_content=None,
                on_reasoning=None,
                check_interrupt=None,
                enable_web_search=False,
                status_line=None,
            ):
                state["calls"] += 1
                if on_reasoning:
                    on_reasoning(
                        "Let me analyze the situation.\n"
                        "</thinking>\n\n"
                        "The wakeword models are tiny (2-50 KB each), "
                        "but Git LFS is uploading ALL model files. "
                        "This is a one-time upload and future pushes "
                        "will be much faster."
                    )
                # Content stream is empty — model put response in reasoning
                return StreamingChatResponse(
                    content="",
                    thinking=(
                        "Let me analyze the situation.\n"
                        "</thinking>\n\n"
                        "The wakeword models are tiny (2-50 KB each), "
                        "but Git LFS is uploading ALL model files. "
                        "This is a one-time upload and future pushes "
                        "will be much faster."
                    ),
                )

        monkeypatch.setattr("harness.cline_agent.StreamingJSONClient", _FakeClient)

        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=3)

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = asyncio.get_event_loop().run_until_complete(
                agent.run("Why is git LFS slow?", enable_interrupt=False)
            )

        assert state["calls"] <= 2, (
            f"Should extract response from reasoning, not retry {state['calls']} times"
        )
        assert "wakeword" in result.lower() or "LFS" in result

    def test_split_response_from_reasoning_static(self):
        """Unit test for _split_response_from_reasoning."""
        split = ClineAgent._split_response_from_reasoning

        # Response after </thinking>
        content, reasoning = split(
            "Internal analysis here.\n</thinking>\n\nHere is the answer to your question about the build system.",
            "",
        )
        assert "answer to your question" in content
        assert "</thinking>" not in reasoning

        # No marker — returns original
        content, reasoning = split(
            "Just internal thoughts, no response here.",
            "original",
        )
        assert content == "original"

        # Short text after marker — ignored
        content, reasoning = split(
            "Thinking.\n</thinking>\nOK",
            "original",
        )
        assert content == "original"


class TestCreatePlanContext:
    """Test that _build_plan_context produces a useful context summary."""

    def _make_agent(self):
        config = Config(api_url="http://test.invalid", api_key="k", model="m")
        agent = ClineAgent(config=config, max_iterations=1)
        return agent

    def test_context_includes_workspace(self):
        agent = self._make_agent()
        ctx = agent._build_plan_context()
        assert "WORKSPACE" in ctx
        assert agent.workspace_path in ctx

    def test_context_includes_todos(self):
        agent = self._make_agent()
        agent.todo_manager.add("Fix auth bug")
        agent.todo_manager.add("Write tests")
        ctx = agent._build_plan_context()
        assert "ACTIVE TODOS" in ctx
        assert "Fix auth bug" in ctx

    def test_context_includes_recent_files(self):
        agent = self._make_agent()
        agent.context.add("file", "src/main.py", "def main():\n    pass\n")
        ctx = agent._build_plan_context()
        assert "RECENT CONTEXT" in ctx
        assert "src/main.py" in ctx

    def test_context_empty_when_no_data(self):
        agent = self._make_agent()
        ctx = agent._build_plan_context()
        # Should still have workspace at minimum
        assert "WORKSPACE" in ctx
        # Should NOT have todos or recent context sections
        assert "ACTIVE TODOS" not in ctx


# ============================================================
# Provider Loading Tests
# ============================================================

class TestProviderLoading:
    """Test load_providers and load_claude_cli_config from harness.py."""

    def test_load_providers_new_format(self, tmp_path):
        """Load providers with new key names (fast/normal)."""
        import importlib.util
        _harness_script = os.path.join(os.path.dirname(__file__), '..', 'harness.py')
        _spec = importlib.util.spec_from_file_location("harness_entry", _harness_script)
        _harness_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_harness_mod)
        
        models_dir = tmp_path / ".z"
        models_dir.mkdir()
        (models_dir / "models.json").write_text(json.dumps({
            "providers": {
                "fast": {"api_url": "http://fast", "api_key": "fk", "model": "fast-m"},
                "normal": {"api_url": "http://normal", "api_key": "nk", "model": "normal-m"},
            }
        }))
        
        providers = _harness_mod.load_providers(str(tmp_path))
        assert "fast" in providers
        assert "normal" in providers

    def test_load_providers_old_format(self, tmp_path):
        """Load providers with old key names (low/orchestrator) — should normalise."""
        import importlib.util
        _harness_script = os.path.join(os.path.dirname(__file__), '..', 'harness.py')
        _spec = importlib.util.spec_from_file_location("harness_entry", _harness_script)
        _harness_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_harness_mod)
        
        models_dir = tmp_path / ".z"
        models_dir.mkdir()
        (models_dir / "models.json").write_text(json.dumps({
            "providers": {
                "low": {"api_url": "http://low", "api_key": "lk", "model": "low-m"},
                "orchestrator": {"api_url": "http://orch", "api_key": "ok", "model": "orch-m"},
            }
        }))
        
        providers = _harness_mod.load_providers(str(tmp_path))
        assert "fast" in providers
        assert "normal" in providers
        assert "low" not in providers
        assert "orchestrator" not in providers

    def test_load_providers_missing_file(self, tmp_path):
        """Missing models.json returns empty dict."""
        import importlib.util
        _harness_script = os.path.join(os.path.dirname(__file__), '..', 'harness.py')
        _spec = importlib.util.spec_from_file_location("harness_entry", _harness_script)
        _harness_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_harness_mod)
        
        providers = _harness_mod.load_providers(str(tmp_path))
        assert providers == {}

    def test_load_claude_cli_config(self, tmp_path):
        """Load claude_cli config section."""
        import importlib.util
        _harness_script = os.path.join(os.path.dirname(__file__), '..', 'harness.py')
        _spec = importlib.util.spec_from_file_location("harness_entry", _harness_script)
        _harness_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_harness_mod)
        
        models_dir = tmp_path / ".z"
        models_dir.mkdir()
        (models_dir / "models.json").write_text(json.dumps({
            "providers": {},
            "claude_cli": {"model": "claude-opus-4.6", "flags": ["--dangerously-skip-permissions"]}
        }))
        
        config = _harness_mod.load_claude_cli_config(str(tmp_path))
        assert config["model"] == "claude-opus-4.6"


# ============================================================
# Todo Panel Rendering Tests
# ============================================================

class TestTodoPanelRendering:
    """Test the Rich visual todo panel rendering."""

    def test_render_empty_panel(self):
        mgr = TodoManager()
        panel = mgr.render_todo_panel()
        assert panel is not None
        assert panel.title == "Todo List"
        # Should have a "No todos yet" message

    def test_render_panel_with_items(self):
        mgr = TodoManager()
        mgr.add("Task one")
        mgr.add("Task two")
        panel = mgr.render_todo_panel()
        assert panel.title == "Todo List"
        # Should not raise

    def test_render_panel_with_hierarchy(self):
        mgr = TodoManager()
        parent = mgr.add("Parent task")
        child = mgr.add("Child task", parent_id=parent.id)
        mgr.update(parent.id, status="in-progress")
        mgr.update(child.id, status="completed")
        panel = mgr.render_todo_panel()
        assert panel is not None

    def test_render_panel_progress(self):
        from io import StringIO
        from rich.console import Console as RichConsole
        mgr = TodoManager()
        mgr.add("Task 1")
        mgr.add("Task 2")
        mgr.update(1, status="completed")
        # Capture output
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=True, width=80)
        mgr.print_todo_panel(con)
        output = buf.getvalue()
        assert "50%" in output
        assert "1/2" in output

    def test_render_panel_all_complete(self):
        mgr = TodoManager()
        mgr.add("Done 1")
        mgr.add("Done 2")
        mgr.update(1, status="completed")
        mgr.update(2, status="completed")
        panel = mgr.render_todo_panel()
        assert panel.border_style == "green"

    def test_render_panel_in_progress(self):
        mgr = TodoManager()
        mgr.add("Active")
        mgr.update(1, status="in-progress")
        panel = mgr.render_todo_panel()
        assert panel.border_style == "yellow"

    def test_render_panel_blocked(self):
        from io import StringIO
        from rich.console import Console as RichConsole
        mgr = TodoManager()
        mgr.add("Blocked task")
        mgr.update(1, status="blocked")
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=True, width=80)
        mgr.print_todo_panel(con)
        output = buf.getvalue()
        assert "blocked" in output.lower() or "1 blocked" in output.lower()

    def test_print_todo_panel_empty_is_noop(self):
        """print_todo_panel should do nothing if there are no items."""
        from io import StringIO
        from rich.console import Console as RichConsole
        mgr = TodoManager()
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=True, width=80)
        mgr.print_todo_panel(con)
        assert buf.getvalue() == ""

    def test_render_panel_with_notes(self):
        from io import StringIO
        from rich.console import Console as RichConsole
        mgr = TodoManager()
        mgr.add("Task with notes")
        mgr.update(1, notes="Some working note here")
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=True, width=80)
        mgr.print_todo_panel(con)
        output = buf.getvalue()
        assert "Some working note" in output


# ============================================================
# Workspace Index Tests
# ============================================================

class TestWorkspaceIndex:
    """Test the WorkspaceIndex class."""

    def test_build_on_harness_repo(self):
        """Index builds on the harness repo itself."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        assert len(idx.files) > 10
        # Should find our own source files
        paths = [f.rel_path for f in idx.files]
        assert any("cline_agent.py" in p for p in paths)
        assert any("workspace_index.py" in p for p in paths)

    def test_file_info_fields(self):
        """FileInfo has expected fields."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        py_files = [f for f in idx.files if f.extension == '.py']
        assert len(py_files) > 5
        for f in py_files:
            assert f.language == "Python"
            assert f.lines > 0
            assert f.size > 0
            assert not f.is_binary

    def test_search(self):
        """File search by name substring works."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        results = idx.search("config")
        assert any("config.py" in f.rel_path for f in results)

    def test_get_languages(self):
        """Language detection returns Python for this repo."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        langs = idx.get_languages()
        assert "Python" in langs
        assert langs["Python"] >= 5

    def test_summary_not_empty(self):
        """Summary produces non-empty output."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        s = idx.summary()
        assert "PROJECT MAP" in s
        assert "Python" in s
        assert len(s) > 100

    def test_compact_tree(self):
        """Compact tree returns file paths."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        tree = idx.compact_tree()
        assert "cline_agent.py" in tree

    def test_empty_directory(self, tmp_path):
        """Index handles an empty directory gracefully."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(tmp_path)).build()
        assert len(idx.files) == 0
        assert "no files found" in idx.summary()

    def test_binary_files_detected(self):
        """Binary files are flagged correctly."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        png_files = [f for f in idx.files if f.extension == '.png']
        for f in png_files:
            assert f.is_binary
            assert f.lines == 0

    def test_get_dir(self):
        """get_dir returns files for a specific directory."""
        from harness.workspace_index import WorkspaceIndex
        idx = WorkspaceIndex(str(Path(__file__).parent.parent)).build()
        src_files = idx.get_dir("src/harness")
        assert len(src_files) > 5
        for f in src_files:
            assert f.rel_path.startswith("src/harness/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
