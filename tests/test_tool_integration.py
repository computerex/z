"""Integration tests for the native tool-use pipeline.

These tests validate the ACTUAL pipeline the agent uses:
  Native tool_calls -> ParsedToolCall -> _execute_tool() -> result

Unlike unit tests which test Python classes in isolation, these prove
that a structured tool call is correctly dispatched, executed, and
reflected in agent state. This is the critical gap that unit tests miss.
"""

import asyncio
import io
import sys
import os
import json
import re
from pathlib import Path
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.cline_agent import (
    ParsedToolCall,
    ClineAgent,
)
from harness.config import Config
from harness.todo_manager import TodoManager, TodoStatus
from harness.context import SmartContextManager
from harness.streaming_client import StreamingChatResponse


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
            parameters={"action": "add", "title": "Fix the auth bug"},
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
                "context_refs": "src/auth.py,src/config.py",
            },
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
            name="manage_todos", parameters={"action": "add", "title": "Main task"}
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool1))

        # Add child
        tool2 = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Sub-task", "parent_id": "1"},
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
            parameters={"action": "update", "id": "1", "status": "in-progress"},
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
                "notes": "Found the bug in auth.py line 42",
            },
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Updated todo [1]" in result
        assert agent.todo_manager.get(1).notes == "Found the bug in auth.py line 42"

    def test_dispatch_manage_todos_complete(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")

        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "1", "status": "completed"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Updated todo [1]" in result
        assert agent.todo_manager.get(1).status == TodoStatus.COMPLETED

    def test_dispatch_manage_todos_remove(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")

        tool = ParsedToolCall(
            name="manage_todos", parameters={"action": "remove", "id": "1"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Removed todo [1]" in result
        assert len(agent.todo_manager.list_all()) == 0

    def test_dispatch_manage_todos_list(self):
        agent = self._make_agent()
        agent.todo_manager.add("Task 1")
        agent.todo_manager.add("Task 2")

        tool = ParsedToolCall(name="manage_todos", parameters={"action": "list"})
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Task 1" in result
        assert "Task 2" in result

    def test_dispatch_manage_todos_error_no_title(self):
        agent = self._make_agent()
        tool = ParsedToolCall(name="manage_todos", parameters={"action": "add"})
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "'title' is required" in result

    def test_dispatch_manage_todos_error_invalid_id(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos", parameters={"action": "update", "id": "abc"}
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "Invalid id" in result

    def test_dispatch_manage_todos_error_not_found(self):
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "99", "status": "completed"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "not found" in result

    def test_dispatch_manage_todos_error_unknown_action(self):
        agent = self._make_agent()
        tool = ParsedToolCall(name="manage_todos", parameters={"action": "destroy"})
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert "Error" in result
        assert "Unknown action" in result

    def test_dispatch_execute_command_missing_command_param(self):
        """Malformed execute_command without <command> should be rejected."""
        agent = self._make_agent()
        tool = ParsedToolCall(
            name="execute_command",
            parameters={},
        )
        result = asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool))
        assert (
            "Error: execute_command requires a non-empty <command> parameter" in result
        )


# ============================================================
# End-to-End Mock Tests — Simulate full model output → parse → execute
# ============================================================


class TestEndToEndMock:
    """Simulate realistic native tool calls and verify the complete pipeline.

    These test what happens in _run_loop when the model emits a tool call:
    1. Native tool_call → ParsedToolCall
    2. ParsedToolCall → _execute_tool() → result string
    3. Result gets appended as a tool message for next iteration
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
            StreamingMessage(
                role="system", content=get_system_prompt(agent.workspace_path)
            )
        ]
        agent._initialized = True
        return agent

    def test_e2e_model_adds_todo_then_reads_file(self):
        """Simulate: model → add todo tool call → execute → verify state."""
        agent = self._make_agent()

        # Native tool call as emitted by the model
        tool_call = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "add",
                "title": "Read and understand auth module",
                "context_refs": "src/auth.py",
            },
        )
        assert tool_call.name == "manage_todos"

        # Execute the tool
        result = asyncio.get_event_loop().run_until_complete(
            agent._execute_tool(tool_call)
        )

        # Verify state changes
        assert "Added todo [1]" in result
        assert len(agent.todo_manager.list_all()) == 1
        item = agent.todo_manager.get(1)
        assert item.title == "Read and understand auth module"
        assert "src/auth.py" in item.context_refs

    def test_e2e_model_creates_full_todo_list(self):
        """Simulate: model creates multiple todos, updates their status."""
        agent = self._make_agent()

        # Step 1: Add first todo
        tool1 = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "add", "title": "Fix authentication bug"},
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool1))

        # Step 2: Add sub-task
        tool2 = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "add",
                "title": "Add null check for token",
                "parent_id": "1",
                "context_refs": "src/auth.py",
            },
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool2))

        # Step 3: Mark parent in-progress
        tool3 = ParsedToolCall(
            name="manage_todos",
            parameters={"action": "update", "id": "1", "status": "in-progress"},
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool3))

        # Step 4: Complete sub-task
        tool4 = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "update",
                "id": "2",
                "status": "completed",
                "notes": "Added null check on line 42",
            },
        )
        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tool4))

        # Verify final state
        assert len(agent.todo_manager.list_all()) == 2
        parent = agent.todo_manager.get(1)
        child = agent.todo_manager.get(2)
        assert parent.status == TodoStatus.IN_PROGRESS
        assert child.status == TodoStatus.COMPLETED
        assert child.parent_id == 1
        assert child.notes == "Added null check on line 42"

    def test_e2e_todo_persists_through_save_load(self):
        """Todos survive session save/load."""
        import tempfile

        agent = self._make_agent()

        # Add todos
        tool = ParsedToolCall(
            name="manage_todos",
            parameters={
                "action": "add",
                "title": "Persistent task",
                "context_refs": "main.py",
            },
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
        from harness.context import CompactionTrace

        agent.smart_context.compaction_traces.append(
            CompactionTrace("file_read", "test.py", "test", tokens_freed=100)
        )

        agent.clear_history()
        assert len(agent.todo_manager.list_all()) == 0
        assert len(agent.smart_context.compaction_traces) == 0

    def test_eviction_count_in_stats(self):
        """Eviction count shows in context stats."""
        agent = self._make_agent()
        from harness.context import CompactionTrace

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
            StreamingMessage(
                role="system", content=get_system_prompt(agent.workspace_path)
            )
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

    def test_dump_includes_full_messages(self):
        import tempfile
        from harness.streaming_client import StreamingMessage

        agent = self._make_agent()
        # Add some messages
        agent.messages.append(
            StreamingMessage(role="user", content="Hello, please read config.py")
        )
        agent.messages.append(
            StreamingMessage(
                role="assistant",
                content="I'll read that file.\n\n<read_file>\n<path>config.py</path>\n</read_file>",
            )
        )
        agent.messages.append(
            StreamingMessage(
                role="user", content="[read_file result]\ndef main():\n    pass"
            )
        )

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
            # System prompt is substantial. Native tool calling moves the tool
            # schemas out of the prompt (into the `tools` param), so this is
            # leaner than the old XML-embedded prompt but still sizeable.
            assert data["messages"][0]["tokens_est"] > 800
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


# ============================================================
# XML Parsing Tests
# ============================================================


# ============================================================
# Todo Panel Rendering Tests
# ============================================================


class TestTodoPanelRendering:
    """Test the Rich visual todo panel rendering."""

    def test_render_empty_panel(self):
        mgr = TodoManager()
        panel = mgr.render_todo_panel()
        assert panel is not None
        assert "Todo" in panel.title
        # Should have a "No todos yet" message

    def test_render_panel_with_items(self):
        mgr = TodoManager()
        mgr.add("Task one")
        mgr.add("Task two")
        panel = mgr.render_todo_panel()
        assert "Todo" in panel.title
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
        py_files = [f for f in idx.files if f.extension == ".py"]
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
        png_files = [f for f in idx.files if f.extension == ".png"]
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
