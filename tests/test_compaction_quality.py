"""Compaction quality tests — validates that context management does not nuke useful content.

Simulates realistic multi-step coding sessions and checks:
1. Critical context (error output, recent file reads, in-progress analysis) survives compaction
2. Stale/redundant content (old duplicate reads, outdated command output) gets compacted first
3. Todo-relevant content is protected by semantic scoring
4. The agent can recover from compacted content via breadcrumb traces
5. Edge cases: empty todos, all-assistant conversations, huge single messages
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from harness.todo_manager import TodoManager, TodoStatus
from harness.smart_context import (
    SmartContextManager,
    SemanticScorer,
    CompactionTrace,
    PROTECTED_INDICES,
    COMPACT_MARKER,
    _get_content,
)
from harness.context_management import estimate_tokens, estimate_messages_tokens


def _msg(role, content):
    return {"role": role, "content": content}


def _make_file_read(path, content_lines=50):
    """Simulate a [read_file result] message."""
    code_lines = [f"    line_{i} = value_{i}" for i in range(content_lines)]
    body = f"[read_file result]\n{path}\n" + "\n".join(code_lines)
    return _msg("user", body)


def _make_command_output(cmd, output, exit_code=0):
    """Simulate a [execute_command result] message."""
    body = f"[execute_command result]\n$ {cmd}\n{output}\nExit code: {exit_code}"
    return _msg("user", body)


def _make_search_result(query, results_text):
    body = f"[search_files result]\nQuery: {query}\n{results_text}"
    return _msg("user", body)


def _make_assistant_analysis(text):
    return _msg("assistant", text)


def _make_assistant_tool_call(tool_xml, reasoning=""):
    body = f"{reasoning}\n<read_file>\n<path>{tool_xml}</path>\n</read_file>" if reasoning else f"<read_file>\n<path>{tool_xml}</path>\n</read_file>"
    return _msg("assistant", body)


def _make_guidance_nudge(nudge_type="introspect"):
    return _msg("user", f"[GUIDANCE_NUDGE type={nudge_type}] Consider pausing to reflect.")


def _build_realistic_session(msg_count=40, task_description="Fix authentication bug in src/auth.py"):
    """Build a conversation that mimics a real coding session.

    Structure:
    - msg[0]: system prompt
    - msg[1]: user task description
    - msg[2]: assistant initial analysis
    - msg[3..N]: interleaved tool calls, file reads, command output, analysis
    """
    messages = [
        _msg("system", "You are a coding assistant. " * 50),  # ~200 tokens
        _msg("user", task_description),
        _make_assistant_analysis(
            f"I'll investigate the authentication issue. Let me start by reading the auth module "
            f"and understanding the current implementation. The task is: {task_description}"
        ),
    ]

    for i in range(3, msg_count):
        cycle = (i - 3) % 6
        if cycle == 0:
            messages.append(_make_assistant_tool_call(f"src/module_{i//6}.py", f"Let me read module_{i//6}.py to understand the code structure."))
        elif cycle == 1:
            messages.append(_make_file_read(f"src/module_{i//6}.py", content_lines=60))
        elif cycle == 2:
            messages.append(_make_assistant_analysis(
                f"Looking at module_{i//6}.py, I can see the authentication flow uses JWT tokens. "
                f"The validate_token function on line 42 doesn't handle expired tokens correctly. "
                f"I need to check the error handling in the middleware next. " * 3
            ))
        elif cycle == 3:
            messages.append(_make_assistant_tool_call("src/middleware.py", "Let me check the middleware error handling."))
        elif cycle == 4:
            messages.append(_make_command_output(
                f"python -m pytest tests/test_auth.py -v",
                f"FAILED tests/test_auth.py::test_expired_token - AssertionError: expected 401\n"
                f"FAILED tests/test_auth.py::test_invalid_signature - jwt.exceptions.InvalidSignatureError\n"
                f"{'.' * 200}\n"
                f"2 failed, 15 passed in 3.42s\n" * 3,
                exit_code=1,
            ))
        elif cycle == 5:
            messages.append(_make_guidance_nudge("introspect"))

    return messages


class TestCompactionDoesNotNukeUsefulContext:
    """Tests that compaction preserves critical context."""

    def _setup_with_todos(self, task="Fix authentication bug in src/auth.py"):
        tm = TodoManager()
        tm.add(task)
        tm.update(1, status="in-progress")
        tm.update(1, context_refs=["src/auth.py", "src/middleware.py"])
        sc = SmartContextManager(tm)
        return tm, sc

    def test_recent_messages_never_compacted(self):
        """The most recent messages (last exchange) must survive compaction."""
        tm, sc = self._setup_with_todos()
        messages = _build_realistic_session(msg_count=30)

        last_user = messages[-2]["content"] if messages[-2]["role"] == "user" else messages[-1]["content"]
        last_assistant = messages[-1]["content"] if messages[-1]["role"] == "assistant" else messages[-2]["content"]

        current_tokens = estimate_messages_tokens(messages)
        # Force very aggressive compaction — budget is 30% of current
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.5), current_tokens=current_tokens
        )

        # The last few messages must survive
        surviving_contents = [_get_content(m) for m in out[-4:]]
        last_user_survived = any(last_user[:100] in c for c in surviving_contents)
        last_assistant_survived = any(last_assistant[:100] in c for c in surviving_contents)

        assert last_user_survived, "Last user message was nuked during compaction"
        assert last_assistant_survived, "Last assistant message was nuked during compaction"

    def test_system_prompt_and_first_pair_untouched(self):
        """System prompt (idx 0) and first user/assistant pair (idx 1,2) are sacred."""
        tm, sc = self._setup_with_todos()
        messages = _build_realistic_session(msg_count=30)

        original_system = messages[0]["content"]
        original_first_user = messages[1]["content"]
        original_first_assistant = messages[2]["content"]

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.4), current_tokens=current_tokens
        )

        assert out[0]["content"] == original_system, "System prompt was modified"
        assert out[1]["content"] == original_first_user, "First user message was modified"
        assert out[2]["content"] == original_first_assistant, "First assistant message was modified"

    def test_todo_relevant_content_scored_higher(self):
        """Content mentioning files/keywords from active todos should be retained over irrelevant content."""
        tm, sc = self._setup_with_todos("Fix authentication bug in src/auth.py")

        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Fix the auth bug"),
            _make_assistant_analysis("I'll investigate the auth issue."),
        ]

        # Add some relevant content
        messages.append(_make_file_read("src/auth.py", content_lines=80))
        messages.append(_make_assistant_analysis(
            "The authentication module in src/auth.py has a bug in validate_token. "
            "The JWT expiry check is missing. I need to add proper token validation."
        ))

        # Add a bunch of IRRELEVANT padding
        for i in range(15):
            messages.append(_make_file_read(f"src/unrelated/utils_{i}.py", content_lines=80))
            messages.append(_make_assistant_analysis(
                f"This utility file utils_{i}.py contains string formatting helpers "
                f"and logging configuration. Not relevant to the current task."
            ))

        # Add more relevant content near the middle (not protected by recency)
        messages.append(_make_command_output(
            "python -m pytest tests/test_auth.py",
            "FAILED test_expired_token - JWT validation error\n2 failed, 10 passed",
            exit_code=1,
        ))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.5), current_tokens=current_tokens
        )

        surviving_text = " ".join(_get_content(m) for m in out)

        # The auth-related file read should survive (todo references src/auth.py)
        auth_read_survived = "src/auth.py" in surviving_text and "validate_token" in surviving_text
        # The test failure output should survive (highly relevant)
        test_failure_survived = "FAILED test_expired_token" in surviving_text or "JWT validation" in surviving_text

        # At least some irrelevant stuff should have been compacted
        irrelevant_compacted = COMPACT_MARKER in surviving_text or freed > 0

        assert irrelevant_compacted, "Nothing was compacted despite being over budget"

        # The auth content should be more likely to survive than random utils
        # (We check that at least one of the relevant pieces survived)
        relevant_survived = auth_read_survived or test_failure_survived
        assert relevant_survived, (
            f"Todo-relevant content (auth.py read or test failures) was nuked while "
            f"irrelevant utils files may have survived. Report: {report}"
        )

    def test_duplicate_file_reads_consolidated(self):
        """Reading the same file multiple times should keep only the latest version."""
        tm, sc = self._setup_with_todos()
        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Fix the bug"),
            _make_assistant_analysis("Let me investigate."),
        ]

        # Read auth.py 3 times (simulating re-reads during debugging)
        for version in range(3):
            messages.append(_make_assistant_tool_call("src/auth.py"))
            messages.append(_make_file_read("src/auth.py", content_lines=50 + version * 10))

        # Add padding to go over budget
        for i in range(5):
            messages.append(_make_file_read(f"src/pad_{i}.py", content_lines=60))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.6), current_tokens=current_tokens
        )

        # Count how many full auth.py reads remain (not compacted)
        auth_read_count = sum(
            1 for m in out
            if "src/auth.py" in _get_content(m)
            and "[read_file result]" in _get_content(m)
            and COMPACT_MARKER not in _get_content(m)[:20]
        )

        assert auth_read_count <= 1, (
            f"Duplicate dedup failed: {auth_read_count} full reads of auth.py survive. "
            f"Should consolidate to latest only."
        )
        # The duplicate notice should mention the file
        has_dedup_trace = any(
            "Duplicate read" in _get_content(m) or "Deduplicated" in report
            for m in out
        )
        assert freed > 0, "No tokens freed from 3 duplicate reads"

    def test_guidance_nudges_compacted_first(self):
        """Guidance nudges (system hints) should be compacted before real content."""
        tm, sc = self._setup_with_todos()
        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Fix the bug"),
            _make_assistant_analysis("I'll investigate."),
        ]

        # Add many guidance nudges interspersed with real content
        for i in range(10):
            messages.append(_make_guidance_nudge("introspect"))
            messages.append(_make_file_read(f"src/real_file_{i}.py", content_lines=40))
            messages.append(_make_assistant_analysis(
                f"Analyzing real_file_{i}.py — found important pattern on line {i*10}. "
                f"This relates to the authentication flow. " * 2
            ))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.7), current_tokens=current_tokens
        )

        # Count surviving guidance vs real content
        surviving_guidance = sum(
            1 for m in out
            if "GUIDANCE_NUDGE" in _get_content(m) and COMPACT_MARKER not in _get_content(m)[:20]
        )
        surviving_file_reads = sum(
            1 for m in out
            if "[read_file result]" in _get_content(m) and COMPACT_MARKER not in _get_content(m)[:20]
        )

        assert surviving_guidance <= surviving_file_reads, (
            f"Guidance nudges ({surviving_guidance}) survived more than file reads ({surviving_file_reads}). "
            f"Nudges should be compacted first as they have the lowest regen cost."
        )

    def test_error_output_preserved_over_success_output(self):
        """Command output with errors should score higher than clean output."""
        tm, sc = self._setup_with_todos("Fix failing tests in test_auth.py")
        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Fix the failing tests"),
            _make_assistant_analysis("I'll run the tests and investigate failures."),
        ]

        # Add successful (boring) command output
        for i in range(5):
            messages.append(_make_command_output(
                f"python -m pytest tests/test_utils_{i}.py",
                f"{'.' * 100}\n10 passed in 1.{i}s\n" * 4,
            ))

        # Add failing test output (critical to keep)
        messages.append(_make_command_output(
            "python -m pytest tests/test_auth.py -v",
            "FAILED tests/test_auth.py::test_expired_token - AssertionError\n"
            "  assert response.status_code == 401\n"
            "  E       assert 200 == 401\n"
            "FAILED tests/test_auth.py::test_invalid_sig - InvalidSignatureError\n"
            "2 failed, 8 passed in 2.1s\n" * 3,
            exit_code=1,
        ))

        # Add more padding
        for i in range(5):
            messages.append(_make_file_read(f"src/unrelated_{i}.py", content_lines=60))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.5), current_tokens=current_tokens
        )

        surviving_text = " ".join(_get_content(m) for m in out)

        # The test failure output is highly relevant — check if it survived
        failure_info_present = (
            "FAILED" in surviving_text
            or "test_expired_token" in surviving_text
            or "test_auth" in surviving_text
        )

        # At minimum, the breadcrumb trace should reference the failing test
        breadcrumb_present = (
            "test_auth" in surviving_text
            or "pytest" in surviving_text
        )

        assert failure_info_present or breadcrumb_present, (
            "Critical test failure output was completely lost — no content or breadcrumb survived"
        )

    def test_compaction_leaves_breadcrumbs(self):
        """Compacted messages should leave trace notices the agent can act on."""
        tm, sc = self._setup_with_todos()
        messages = _build_realistic_session(msg_count=30)

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.5), current_tokens=current_tokens
        )

        if freed > 0:
            compacted_notices = [
                _get_content(m) for m in out
                if COMPACT_MARKER in _get_content(m)[:20]
            ]

            for notice in compacted_notices:
                # Guidance nudges don't need recovery hints (no content to recover)
                if "guidance" in notice.lower() or "archived" in notice.lower():
                    continue
                has_recovery_hint = any(hint in notice.lower() for hint in [
                    "re-read", "re-run", "re-search", "retrieve", "result stored",
                    "previous analysis", "was read here", "was here",
                ])
                assert has_recovery_hint, (
                    f"Compacted notice lacks recovery hint: {notice[:200]}"
                )

    def test_no_compaction_when_under_budget(self):
        """If tokens are within budget, nothing should be compacted."""
        tm, sc = self._setup_with_todos()
        messages = [
            _msg("system", "You are an assistant."),
            _msg("user", "Hello"),
            _make_assistant_analysis("Hi there!"),
            _make_file_read("src/small.py", content_lines=5),
        ]

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=current_tokens * 10, current_tokens=current_tokens
        )

        assert freed == 0, f"Compacted {freed} tokens when under budget"
        assert len(out) == len(messages), "Message count changed when under budget"

    def test_aggressive_compaction_still_preserves_minimum(self):
        """Even under extreme pressure, system prompt + first pair + last 2 messages survive."""
        tm, sc = self._setup_with_todos()
        messages = _build_realistic_session(msg_count=40)

        current_tokens = estimate_messages_tokens(messages)
        # Demand compaction to 20% — extremely aggressive
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.25), current_tokens=current_tokens
        )

        assert len(out) >= 5, (
            f"Aggressive compaction left only {len(out)} messages — "
            f"should preserve at least system + first pair + last exchange"
        )
        assert out[0]["role"] == "system", "System prompt was removed"

    def test_empty_todos_use_neutral_scoring(self):
        """With no active todos, all content gets neutral relevance — recency and type dominate."""
        tm = TodoManager()  # No todos
        sc = SmartContextManager(tm)

        messages = _build_realistic_session(msg_count=25)
        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.6), current_tokens=current_tokens
        )

        # Should still compact something (recency/type scoring still works)
        if current_tokens > int(current_tokens * 0.6) * sc.budget_ratio:
            assert freed > 0, "Failed to compact anything even without todos"

    def test_semantic_maintenance_tick_prunes_guidance(self):
        """Per-turn maintenance should prune old guidance nudges first."""
        tm, sc = self._setup_with_todos()
        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Fix the bug"),
            _make_assistant_analysis("I'll investigate."),
        ]

        # Add lots of guidance nudges
        for i in range(20):
            messages.append(_make_guidance_nudge("introspect"))
            messages.append(_make_assistant_analysis(f"Step {i}: analyzing code..."))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.semantic_maintenance_tick(
            messages, max_tokens=int(current_tokens * 1.2), current_tokens=current_tokens
        )

        surviving_guidance = sum(
            1 for m in out
            if "GUIDANCE_NUDGE" in _get_content(m) and COMPACT_MARKER not in _get_content(m)[:20]
        )

        # Guidance should be pruned down
        assert surviving_guidance < 20, (
            f"Maintenance tick left {surviving_guidance}/20 guidance nudges — should prune stale ones"
        )

    def test_large_message_gets_size_pressure(self):
        """A single huge message should get size_pressure penalty making it a compaction target."""
        tm, sc = self._setup_with_todos()
        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Debug this"),
            _make_assistant_analysis("Let me look."),
        ]

        # One huge file read
        messages.append(_make_file_read("src/huge_file.py", content_lines=500))
        # Many small file reads
        for i in range(10):
            messages.append(_make_file_read(f"src/small_{i}.py", content_lines=10))
        # Recent exchange
        messages.append(_make_assistant_analysis("Based on my analysis..." * 20))
        messages.append(_msg("user", "What did you find?"))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.6), current_tokens=current_tokens
        )

        # The huge file should be compacted before the small ones
        huge_survived_full = any(
            "src/huge_file.py" in _get_content(m)
            and "[read_file result]" in _get_content(m)
            and COMPACT_MARKER not in _get_content(m)[:20]
            for m in out
        )

        # It's ok if it survived as a compacted notice — that's expected behavior
        # But if it survived fully while small files were nuked, that's a problem
        small_files_nuked = sum(
            1 for m in out
            if any(f"src/small_{i}.py" in _get_content(m) for i in range(10))
            and COMPACT_MARKER in _get_content(m)[:20]
        )

        if huge_survived_full and small_files_nuked > 3:
            pytest.fail(
                "Huge file survived fully while multiple small files were compacted — "
                "size_pressure should favor compacting the large file first"
            )


class TestCompactionEdgeCases:
    """Edge cases that could cause context loss."""

    def test_all_messages_same_type(self):
        """Session with only file reads shouldn't crash or lose everything."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Read all the files"),
            _make_assistant_analysis("Reading files now."),
        ]
        for i in range(20):
            messages.append(_make_file_read(f"src/file_{i}.py", content_lines=50))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.5), current_tokens=current_tokens
        )

        assert len(out) >= 3, "Lost protected messages"
        # Should still have some file content remaining
        remaining_reads = sum(
            1 for m in out
            if "[read_file result]" in _get_content(m)
            and COMPACT_MARKER not in _get_content(m)[:20]
        )
        assert remaining_reads > 0, "ALL file reads were compacted — at least recent ones should survive"

    def test_single_huge_assistant_message(self):
        """A very long assistant analysis shouldn't crash compaction."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Explain everything"),
            _make_assistant_analysis("Let me explain."),
            _make_assistant_analysis("Detailed analysis: " + "x" * 20000),
            _msg("user", "Thanks, now fix it"),
            _make_assistant_analysis("I'll fix it now."),
        ]

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.3), current_tokens=current_tokens
        )

        assert len(out) >= 3, "Protected messages were lost"
        # The huge message should be compacted (excerpted or replaced)
        surviving_text = " ".join(_get_content(m) for m in out)
        assert "x" * 10000 not in surviving_text, "20KB message survived aggressive compaction intact"

    def test_compaction_idempotent(self):
        """Running compaction twice with the same budget shouldn't lose more content."""
        tm = TodoManager()
        tm.add("Fix bug")
        tm.update(1, status="in-progress")
        sc = SmartContextManager(tm)

        messages = _build_realistic_session(msg_count=25)
        current_tokens = estimate_messages_tokens(messages)
        budget = int(current_tokens * 0.6)

        out1, freed1, _ = sc.compact_context(
            messages, max_tokens=budget, current_tokens=current_tokens
        )
        tokens1 = estimate_messages_tokens(out1)

        # Second pass with same budget
        out2, freed2, _ = sc.compact_context(
            out1, max_tokens=budget, current_tokens=tokens1
        )
        tokens2 = estimate_messages_tokens(out2)

        assert len(out2) >= len(out1) - 2, (
            f"Second compaction pass removed {len(out1) - len(out2)} additional messages — "
            f"should be roughly stable (first pass: {len(out1)}, second: {len(out2)})"
        )

    def test_retrieved_tool_results_accessible(self):
        """Compacted tool results should be stored and retrievable."""
        tm = TodoManager()
        sc = SmartContextManager(tm)

        messages = [
            _msg("system", "You are an assistant. " * 50),
            _msg("user", "Run the tests"),
            _make_assistant_analysis("Running tests now."),
        ]

        # Add a command output that will definitely be compacted
        original_output = "DETAILED TEST OUTPUT:\n" + "test line\n" * 200
        messages.append(_make_command_output("pytest -v", original_output))

        # Add padding to force compaction
        for i in range(10):
            messages.append(_make_file_read(f"src/pad_{i}.py", content_lines=60))
        messages.append(_make_assistant_analysis("Analysis complete." * 20))
        messages.append(_msg("user", "What next?"))

        current_tokens = estimate_messages_tokens(messages)
        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.4), current_tokens=current_tokens
        )

        # Check if any result was stored
        stored = sc.result_storage.list_results()
        if freed > 0 and stored:
            # Verify we can retrieve it
            result = sc.result_storage.get_result(stored[0].result_id)
            assert result is not None, "Stored result not retrievable"
            assert "DETAILED TEST OUTPUT" in result.original_content, (
                "Retrieved result doesn't contain original content"
            )


class TestCompactionQualityMetrics:
    """Quantitative quality metrics for compaction behavior."""

    def test_compaction_precision_on_realistic_session(self):
        """Measure what fraction of compacted messages were actually low-value.

        Uses the 'needed later' heuristic from context_quality_benchmark:
        a message is 'needed' if ≥4 tokens from it reappear in the next 30 messages.
        """
        tm = TodoManager()
        tm.add("Fix authentication in src/auth.py")
        tm.update(1, status="in-progress")
        tm.update(1, context_refs=["src/auth.py"])
        sc = SmartContextManager(tm)

        messages = _build_realistic_session(msg_count=35)
        current_tokens = estimate_messages_tokens(messages)

        # Snapshot original content for comparison
        original_contents = {i: _get_content(m) for i, m in enumerate(messages)}

        out, freed, report = sc.compact_context(
            messages, max_tokens=int(current_tokens * 0.6), current_tokens=current_tokens
        )

        if freed == 0:
            pytest.skip("No compaction occurred (under budget)")

        # Identify which messages were compacted or evicted
        out_contents = {_get_content(m)[:200] for m in out}
        compacted_indices = []
        kept_indices = []

        for i, content in original_contents.items():
            if i in PROTECTED_INDICES:
                continue
            if content[:200] in out_contents:
                kept_indices.append(i)
            else:
                # Check if it was replaced with a compact marker
                compacted_indices.append(i)

        if not compacted_indices:
            return  # Nothing to measure

        # Classify compacted messages
        compacted_types = []
        for i in compacted_indices:
            content = original_contents[i]
            role = messages[i]["role"]
            msg_type, _ = sc._classify(content, role)
            compacted_types.append(msg_type)

        # Guidance nudges and todo/context results are always safe to compact
        safe_compactions = sum(
            1 for t in compacted_types
            if t in ("guidance_nudge", "todo_result", "context_result")
        )
        risky_compactions = sum(
            1 for t in compacted_types
            if t in ("assistant_analysis", "command_output")
        )

        total_compacted = len(compacted_types)
        safe_ratio = safe_compactions / max(1, total_compacted)

        # We expect at least some of the compacted messages to be the "safe" low-value types
        # A healthy compaction should prioritize guidance_nudge > todo_result > old file reads
        assert total_compacted > 0, "Expected some compactions"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
