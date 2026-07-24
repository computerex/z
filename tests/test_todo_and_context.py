"""Tests for TodoManager and SmartContextManager."""

import time
import json
import pytest
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.todo_manager import TodoManager, TodoItem, TodoStatus
from harness.context.context_management import DuplicateDetector
from harness.context.smart_context import SmartContextManager, CompactionTrace, COMPACT_MARKER


# ============================================================
# TodoManager Tests
# ============================================================


class TestTodoManager:
    def test_add_todo(self):
        mgr = TodoManager()
        item = mgr.add("Fix the bug")
        assert item.id == 1
        assert item.title == "Fix the bug"
        assert item.status == TodoStatus.NOT_STARTED

    def test_add_multiple(self):
        mgr = TodoManager()
        t1 = mgr.add("Task 1")
        t2 = mgr.add("Task 2")
        assert t1.id == 1
        assert t2.id == 2
        assert len(mgr.list_all()) == 2

    def test_add_subtask(self):
        mgr = TodoManager()
        parent = mgr.add("Parent task")
        child = mgr.add("Sub-task", parent_id=parent.id)
        assert child.parent_id == parent.id

    def test_update_status(self):
        mgr = TodoManager()
        item = mgr.add("Task")
        mgr.update(item.id, status="in-progress")
        assert mgr.get(item.id).status == TodoStatus.IN_PROGRESS

        mgr.update(item.id, status="completed")
        assert mgr.get(item.id).status == TodoStatus.COMPLETED
        assert mgr.get(item.id).completed_at is not None

    def test_update_notes(self):
        mgr = TodoManager()
        item = mgr.add("Task")
        mgr.update(item.id, notes="Found the issue in auth.py")
        assert mgr.get(item.id).notes == "Found the issue in auth.py"

    def test_update_context_refs(self):
        mgr = TodoManager()
        item = mgr.add("Task")
        mgr.update(item.id, context_refs=["src/auth.py", "src/config.py"])
        assert "src/auth.py" in mgr.get(item.id).context_refs

    def test_remove(self):
        mgr = TodoManager()
        item = mgr.add("Task")
        assert mgr.remove(item.id)
        assert mgr.get(item.id) is None
        assert len(mgr.list_all()) == 0

    def test_remove_with_children(self):
        mgr = TodoManager()
        parent = mgr.add("Parent")
        child = mgr.add("Child", parent_id=parent.id)
        mgr.remove(parent.id)
        assert mgr.get(parent.id) is None
        assert mgr.get(child.id) is None

    def test_list_active(self):
        mgr = TodoManager()
        t1 = mgr.add("Active task")
        t2 = mgr.add("Done task")
        mgr.update(t2.id, status="completed")
        active = mgr.list_active()
        assert len(active) == 1
        assert active[0].id == t1.id

    def test_list_in_progress(self):
        mgr = TodoManager()
        t1 = mgr.add("Not started")
        t2 = mgr.add("In progress")
        mgr.update(t2.id, status="in-progress")
        in_progress = mgr.list_in_progress()
        assert len(in_progress) == 1
        assert in_progress[0].id == t2.id

    def test_get_active_context_refs(self):
        mgr = TodoManager()
        mgr.add("Task 1", context_refs=["file1.py"])
        t2 = mgr.add("Task 2", context_refs=["file2.py"])
        mgr.update(t2.id, status="completed")
        refs = mgr.get_active_context_refs()
        assert "file1.py" in refs
        assert "file2.py" not in refs

    def test_progress_summary(self):
        mgr = TodoManager()
        mgr.add("Task 1")
        t2 = mgr.add("Task 2")
        mgr.update(t2.id, status="in-progress")
        t3 = mgr.add("Task 3")
        mgr.update(t3.id, status="completed")

        summary = mgr.get_progress_summary()
        assert "1/3 complete" in summary
        assert "1 in progress" in summary

    def test_format_list(self):
        mgr = TodoManager()
        mgr.set_original_request("Build the feature")
        mgr.add("Step 1")
        mgr.add("Step 2")

        text = mgr.format_list()
        assert "TODO LIST:" in text
        assert "Step 1" in text
        assert "Step 2" in text
        assert "Build the feature" in text

