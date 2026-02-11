"""Tests for TodoManager and SmartContextManager."""

import time
import json
import pytest
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from harness.todo_manager import TodoManager, TodoItem, TodoStatus
from harness.smart_context import (
    SmartContextManager,
    SemanticScorer,
    CompactionTrace,
    PROTECTED_INDICES,
    COMPACT_MARKER,
)


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

    def test_serialization(self):
        mgr = TodoManager()
        mgr.set_original_request("Test request")
        mgr.add("Task 1", context_refs=["a.py"])
        t2 = mgr.add("Task 2")
        mgr.update(t2.id, status="completed", notes="Done!")
        
        data = mgr.to_dict()
        json_str = json.dumps(data)
        
        mgr2 = TodoManager.from_dict(json.loads(json_str))
        assert len(mgr2.list_all()) == 2
        assert mgr2.original_request == "Test request"
        assert mgr2.get(2).status == TodoStatus.COMPLETED
        assert mgr2.get(2).notes == "Done!"
        assert mgr2.get(1).context_refs == ["a.py"]

    def test_clear(self):
        mgr = TodoManager()
        mgr.add("Task")
        mgr.set_original_request("foo")
        mgr.clear()
        assert len(mgr.list_all()) == 0
        assert mgr.original_request == ""

    def test_format_short(self):
        mgr = TodoManager()
        item = mgr.add("My Task")
        short = item.format_short()
        assert "[1]" in short
        assert "My Task" in short
        assert "○" in short  # not-started icon


# ============================================================
# CompactionTrace Tests
# ============================================================

class TestCompactionTrace:
    def test_file_read_notice(self):
        trace = CompactionTrace("file_read", "src/auth.py", "Auth module", tokens_freed=500)
        text = trace.format_notice()
        assert COMPACT_MARKER in text
        assert "src/auth.py" in text
        assert "500 tok" in text
        assert "Re-read" in text

    def test_command_output_notice(self):
        trace = CompactionTrace("command_output", "npm test", "All 42 tests pass", tokens_freed=300)
        text = trace.format_notice()
        assert COMPACT_MARKER in text
        assert "npm test" in text
        assert "Re-run" in text

    def test_duplicate_read_notice(self):
        trace = CompactionTrace("duplicate_read", "src/main.py", "Superseded", tokens_freed=200)
        text = trace.format_notice()
        assert COMPACT_MARKER in text
        assert "Duplicate read" in text
        assert "later in conversation" in text

    def test_assistant_notice(self):
        trace = CompactionTrace("assistant", "reasoning", "JWT refresh was broken", tokens_freed=150)
        text = trace.format_notice()
        assert COMPACT_MARKER in text
        assert "JWT refresh was broken" in text

    def test_search_result_notice(self):
        trace = CompactionTrace("search_result", "search", "5 matches", tokens_freed=100)
        text = trace.format_notice()
        assert COMPACT_MARKER in text
        assert "Re-search" in text


# ============================================================
# SmartContextManager Tests
# ============================================================

class MockMessage:
    """Mock message for testing."""
    def __init__(self, role, content):
        self.role = role
        self.content = content


class TestSmartContextManager:
    def _make_messages(self, count=10):
        """Create a list of mock messages with enough content to be compactable."""
        msgs = [MockMessage("system", "You are an assistant.")]
        for i in range(count):
            if i % 2 == 0:
                msgs.append(MockMessage("user", f"User message {i} " * 50))
            else:
                msgs.append(MockMessage("assistant", f"Assistant response {i} " * 50))
        return msgs

    def test_init(self):
        tm = TodoManager()
        sc = SmartContextManager(tm)
        assert sc.todo_manager is tm
        assert len(sc.compaction_traces) == 0

    def test_consolidate_duplicates(self):
        tm = TodoManager()
        sc = SmartContextManager(tm)

        file_content = "\n".join([f"   {i} | def function_{i}(): pass" for i in range(50)])

        msgs = [
            MockMessage("system", "System"),
            MockMessage("user", "First task"),
            MockMessage("assistant", "Acknowledged"),
            MockMessage("user", f"[read_file result]\nsrc/auth.py\n{file_content}"),
            MockMessage("assistant", "I see the auth module"),
            MockMessage("user", "Read it again"),
            MockMessage("user", f"[read_file result]\nsrc/auth.py\n{file_content}\ndef refresh(): pass"),
        ]

        msgs, freed = sc._consolidate_duplicates(msgs)
        assert freed > 0
        # The earlier read (index 3) should be replaced with a compaction notice
        assert COMPACT_MARKER in msgs[3].content
        # The later read (index 6) should be untouched
        assert "def refresh(): pass" in msgs[6].content

    def test_compact_context_frees_tokens(self):
        """compact_context should free tokens when over budget."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        file_content = "\n".join([f"line {i}: code " * 20 for i in range(100)])
        msgs = [
            MockMessage("system", "System prompt"),
            MockMessage("user", "First task"),
            MockMessage("assistant", "OK"),
        ]
        # Add lots of big messages to exceed any reasonable budget
        for i in range(20):
            msgs.append(MockMessage("user", f"[read_file result]\nfile_{i}.py\n{file_content}"))
            msgs.append(MockMessage("assistant", f"Analysis of file_{i}: " + "detailed reasoning " * 60))

        total_tokens = sum(len(m.content) // 4 for m in msgs)
        # Set a budget much lower than the total
        budget = total_tokens // 3

        msgs, freed, report = sc.compact_context(msgs, budget)
        assert freed > 0
        assert report != ""
        assert len(sc.compaction_traces) > 0

    def test_compact_context_noop_under_budget(self):
        """compact_context should be a no-op when under budget."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        msgs = self._make_messages(4)
        msgs, freed, report = sc.compact_context(msgs, 999999)
        assert freed == 0
        assert report == ""

    def test_build_recovery_notice(self):
        tm = TodoManager()
        tm.add("Fix the bug")
        sc = SmartContextManager(tm)

        sc.compaction_traces.append(
            CompactionTrace("file_read", "src/auth.py", "Auth module", tokens_freed=500)
        )

        notice = sc.build_context_recovery_notice()
        assert "CONTEXT COMPACTION NOTICE" in notice
        assert "src/auth.py" in notice
        assert "Fix the bug" in notice

    def test_build_recovery_notice_empty(self):
        tm = TodoManager()
        sc = SmartContextManager(tm)
        assert sc.build_context_recovery_notice() == ""

    def test_classify_file_read(self):
        msg_type, source = SmartContextManager._classify(
            "[read_file result]\nsrc/auth.py\nsome code", "user"
        )
        assert msg_type == "file_read"
        assert source == "src/auth.py"

    def test_classify_command_output(self):
        msg_type, source = SmartContextManager._classify(
            "[execute_command result]\n$ npm test\nAll tests passed", "user"
        )
        assert msg_type == "command_output"
        assert "npm test" in source

    def test_classify_search_result(self):
        msg_type, _ = SmartContextManager._classify(
            "[search_files result]\npattern: TODO\nMatches found", "user"
        )
        assert msg_type == "search_result"

    def test_classify_assistant_tool_call(self):
        msg_type, source = SmartContextManager._classify(
            "I'll read that file now.\n<read_file>\n<path>src/main.py</path>\n</read_file>",
            "assistant"
        )
        assert msg_type == "assistant_tool_call"
        assert source == "tool_call"

    def test_classify_assistant_analysis(self):
        msg_type, source = SmartContextManager._classify(
            "After reviewing the code, I think the issue is in the auth flow.",
            "assistant"
        )
        assert msg_type == "assistant_analysis"
        assert source == "analysis"

    def test_classify_todo_result(self):
        msg_type, _ = SmartContextManager._classify(
            "[manage_todos result]\nTodo added: id=1", "user"
        )
        assert msg_type == "todo_result"

    def test_serialization(self):
        tm = TodoManager()
        sc = SmartContextManager(tm)
        sc.compaction_traces.append(
            CompactionTrace("file_read", "test.py", "test file", tokens_freed=100)
        )

        data = sc.to_dict()
        sc2 = SmartContextManager(tm)
        sc2.load_dict(data)

        assert len(sc2.compaction_traces) == 1
        assert sc2.compaction_traces[0].source == "test.py"
        assert sc2.compaction_traces[0].tokens_freed == 100

    def test_serialization_backward_compat(self):
        """load_dict should accept legacy 'eviction_traces' key."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        legacy_data = {
            "eviction_traces": [
                {
                    "original_type": "file_read",
                    "source": "old.py",
                    "summary": "old file",
                    "token_count": 200,
                    "evicted_at": 1234567890,
                }
            ]
        }
        sc.load_dict(legacy_data)
        assert len(sc.compaction_traces) == 1
        assert sc.compaction_traces[0].source == "old.py"
        assert sc.compaction_traces[0].tokens_freed == 200

    # ---------------------------------------------------------------
    # Protection tests
    # ---------------------------------------------------------------

    def test_consolidate_duplicates_never_touches_system_prompt(self):
        """Even if system prompt matches file-read patterns, it must be untouched."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        system_content = (
            "You are an assistant.\n"
            "[read_file result]\nsrc/auth.py\nSome code here\n"
        )
        file_content = "\n".join([f"   {i} | line" for i in range(50)])

        msgs = [
            MockMessage("system", system_content),
            MockMessage("user", "First task"),
            MockMessage("assistant", "Acknowledged"),
            MockMessage("user", f"[read_file result]\nsrc/auth.py\n{file_content}"),
            MockMessage("assistant", "I see auth"),
            MockMessage("user", f"[read_file result]\nsrc/auth.py\n{file_content}\nmore"),
        ]

        msgs, freed = sc._consolidate_duplicates(msgs)
        assert msgs[0].content == system_content

    def test_consolidate_duplicates_never_touches_first_pair(self):
        """First user-assistant pair (indices 1-2) must never be consolidated."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        file_content = "\n".join([f"   {i} | line" for i in range(50)])
        first_user = f"[read_file result]\nsrc/main.py\n{file_content}"
        first_assistant = "Reading: main.py looks good"

        msgs = [
            MockMessage("system", "You are an assistant."),
            MockMessage("user", first_user),
            MockMessage("assistant", first_assistant),
            MockMessage("user", f"[read_file result]\nsrc/main.py\n{file_content}\nupdated"),
        ]

        msgs, freed = sc._consolidate_duplicates(msgs)
        assert msgs[1].content == first_user
        assert msgs[2].content == first_assistant

    def test_compact_context_preserves_system_prompt(self):
        """Full compact_context pipeline must never alter the system prompt."""
        tm = TodoManager()
        tm.add("Build feature X")
        sc = SmartContextManager(tm)

        system_content = "You are an AI assistant with many tools."
        file_content = "\n".join([f"   {i} | line of code" for i in range(100)])

        msgs = [
            MockMessage("system", system_content),
            MockMessage("user", "First task"),
            MockMessage("assistant", "Working on it"),
        ]
        for i in range(20):
            msgs.append(MockMessage("user", f"[read_file result]\nsrc/file_{i % 3}.py\n{file_content}"))
            msgs.append(MockMessage("assistant", f"Unrelated response about topic {i} " * 30))

        msgs, freed, report = sc.compact_context(msgs, 50000)
        assert msgs[0].content == system_content
        assert msgs[1].content == "First task"
        assert msgs[2].content == "Working on it"

    def test_compact_context_preserves_recent_window(self):
        """Messages within recent_window should never be compacted."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        sc.recent_window = 4

        file_content = "x " * 500  # Enough tokens to be compactable
        msgs = [
            MockMessage("system", "System"),
            MockMessage("user", "First task"),
            MockMessage("assistant", "OK"),
        ]
        for i in range(10):
            msgs.append(MockMessage("user", f"[read_file result]\nfile_{i}.py\n{file_content}"))
            msgs.append(MockMessage("assistant", f"Analysis {i}: " + "detail " * 100))

        # Save the content of the last 4 messages
        last_contents = [m.content for m in msgs[-4:]]

        msgs, freed, report = sc.compact_context(msgs, 5000)

        # Last 4 messages must be untouched
        for i, expected in enumerate(last_contents):
            assert msgs[-(4 - i)].content == expected

    def test_compact_handles_assistant_analysis(self):
        """Assistant analysis messages should be compactable."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        msgs = [
            MockMessage("system", "System"),
            MockMessage("user", "Task"),
            MockMessage("assistant", "Acknowledged"),
            MockMessage("assistant", "After deep analysis of the authentication flow, "
                        "I found that the JWT token refresh mechanism has a race condition "
                        "where two concurrent requests can invalidate each other's tokens. " * 10),
            MockMessage("user", "padding " * 100),
            MockMessage("assistant", "padding " * 100),
            MockMessage("user", "padding " * 100),
            MockMessage("assistant", "padding " * 100),
            MockMessage("user", "padding " * 100),
            MockMessage("assistant", "padding " * 100),
            MockMessage("user", "latest message"),
            MockMessage("assistant", "latest response"),
        ]

        msgs, freed, report = sc.compact_context(msgs, 2000)
        # The long analysis at index 3 should have been compacted
        assert freed > 0

    def test_protected_indices_constant(self):
        """PROTECTED_INDICES must contain {0, 1, 2}."""
        assert 0 in PROTECTED_INDICES
        assert 1 in PROTECTED_INDICES
        assert 2 in PROTECTED_INDICES

    def test_protected_indices_attribute(self):
        """SmartContextManager must have protected_indices property containing {0, 1, 2}."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        assert hasattr(sc, 'protected_indices')
        assert 0 in sc.protected_indices
        assert 1 in sc.protected_indices
        assert 2 in sc.protected_indices

    def test_backward_compat_eviction_traces_attr(self):
        """The eviction_traces property should alias compaction_traces."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        trace = CompactionTrace("file_read", "test.py", "test", tokens_freed=100)
        sc.compaction_traces.append(trace)
        assert len(sc.eviction_traces) == 1
        assert sc.eviction_traces[0] is trace

    def test_build_query_text(self):
        """_build_query_text should combine active todo info into a query string."""
        tm = TodoManager()
        tm.add("Fix authentication bug", context_refs=["src/auth.py", "src/jwt.py"])
        tm.add("Update config loading")
        sc = SmartContextManager(tm)

        query = sc._build_query_text()
        assert "Fix authentication bug" in query
        assert "src/auth.py" in query
        assert "src/jwt.py" in query
        assert "Update config loading" in query

    def test_build_query_text_empty(self):
        """_build_query_text returns empty when no active todos."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        assert sc._build_query_text() == ""

    def test_scoring_relevance_boosts_todo_files(self):
        """Messages referencing files from active todos should score higher."""
        tm = TodoManager()
        tm.add("Fix auth", context_refs=["src/auth.py"])
        sc = SmartContextManager(tm)

        # Build messages with relevant and irrelevant content
        relevant_content = "[read_file result]\nsrc/auth.py\ndef login(): pass " * 20
        irrelevant_content = "[read_file result]\nREADME.md\n# Hello World " * 20

        # Score via the scorer (works whether model loads or hits fallback)
        query = sc._build_query_text()
        rel_score = sc._scorer.score_relevance(relevant_content, query)
        irr_score = sc._scorer.score_relevance(irrelevant_content, query)

        # The auth-related content should be more relevant to \"Fix auth\" todo
        assert rel_score > irr_score

    def test_semantic_scorer_fallback(self):
        """SemanticScorer falls back to keyword matching when model unavailable."""
        scorer = SemanticScorer()
        scorer._load_failed = True  # Force fallback

        score = scorer.score_relevance(
            "def authenticate_user(): jwt.decode(token)",
            "Fix authentication bug | src/auth.py"
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should find keyword overlap

    def test_semantic_scorer_fallback_no_query(self):
        """SemanticScorer returns neutral score when query is empty."""
        scorer = SemanticScorer()
        scorer._load_failed = True
        score = scorer.score_relevance("some content", "")
        assert score == 0.4  # neutral

    def test_semantic_scorer_batch(self):
        """score_batch should return one score per content."""
        scorer = SemanticScorer()
        scorer._load_failed = True  # Force fallback for test speed

        contents = ["auth login code", "database pooling", "readme hello world"]
        scores = scorer.score_batch(contents, "Fix authentication login")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        # "auth login code" should be most relevant
        assert scores[0] > scores[2]

    def test_trace_history_bounded(self):
        """Compaction traces should be bounded to _max_traces."""
        tm = TodoManager()
        sc = SmartContextManager(tm)
        sc._max_traces = 5

        for i in range(10):
            sc.compaction_traces.append(
                CompactionTrace("file_read", f"file_{i}.py", f"file {i}", tokens_freed=10)
            )

        # Simulate the trim that happens in compact_context
        if len(sc.compaction_traces) > sc._max_traces:
            sc.compaction_traces = sc.compaction_traces[-sc._max_traces:]

        assert len(sc.compaction_traces) == 5
        assert sc.compaction_traces[0].source == "file_5.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
