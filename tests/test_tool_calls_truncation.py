"""Test that truncate_conversation preserves tool_calls/tool message group integrity."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataclasses import dataclass
from harness.context.context_management import truncate_conversation, _is_tool, _has_tool_calls, _tc_count


@dataclass
class Msg:
    role: str
    content: str
    tool_calls: list = None
    tool_call_id: str = None


def test_imports():
    """Helper functions exist and work."""
    assert not _is_tool(Msg("assistant", "hi"))
    assert _is_tool(Msg("tool", "result"))
    assert not _has_tool_calls(Msg("assistant", "hi"))
    assert _has_tool_calls(Msg("assistant", "", tool_calls=[{"id": "1"}]))
    assert _tc_count(Msg("assistant", "", tool_calls=[{"id": "1"}, {"id": "2"}])) == 2
    assert _tc_count(Msg("assistant", "hi")) == 0


def test_under_four_messages_noop():
    """Fewer than 4 messages is a no-op."""
    msgs = [Msg("system", "s"), Msg("user", "u"), Msg("assistant", "a")]
    result = truncate_conversation(msgs, "half")
    assert len(result.messages) == 3
    assert result.removed_count == 0


def test_orphaned_tool_dropped():
    """Tool messages at the start of kept_messages are orphaned and dropped."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),  # removed
        Msg("assistant", "a2"),  # removed
        Msg("user", "u3"),  # kept (tail)
        Msg("assistant", "tc", tool_calls=[{"id": "1"}]),  # kept
        Msg("tool", "r1", tool_call_id="1"),  # kept
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool"], f"Got: {roles}"


def test_orphaned_tool_calls_dropped_on_missing_results():
    """Assistant with tool_calls but insufficient tool results → whole group dropped."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),  # removed
        Msg("assistant", "a2"),  # removed
        Msg("user", "u3"),  # kept (tail)
        Msg("assistant", "tc", tool_calls=[{"id": "1"}, {"id": "2"}]),  # 2 calls
        Msg("tool", "r1", tool_call_id="1"),  # only 1 result
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert "tool" not in roles, f"Orphaned tool found: {roles}"
    for m in result.messages:
        assert not getattr(m, 'tool_calls', None), f"Orphaned assistant(tc): {m}"


def test_complete_tool_group_preserved():
    """Complete tool_calls+tool group in the tail is preserved when it fits."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u_rm1"),
        Msg("assistant", "a_rm1"),
        Msg("user", "u_rm2"),
        Msg("assistant", "a_rm2"),
        Msg("user", "u_rm3"),
        Msg("assistant", "tc", tool_calls=[{"id": "1"}, {"id": "2"}]),
        Msg("tool", "r1", tool_call_id="1"),
        Msg("tool", "r2", tool_call_id="2"),
        Msg("user", "u_final"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert roles == ["system", "user", "assistant", "user",
                     "assistant", "tool", "tool", "user"], f"Got: {roles}"
    assert sum(1 for r in roles if r == "tool") == 2


def test_dict_messages_work():
    """Truncation also works with plain dict messages."""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "tc", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "1"},
    ]
    result = truncate_conversation(msgs, "half")
    roles = []
    for m in result.messages:
        if hasattr(m, 'role'):
            roles.append(m.role)
        elif isinstance(m, dict):
            roles.append(m.get('role', ''))
    assert len(result.messages) == 6, f"Expected 6: {roles}"
    assert "tool" in roles


def test_notice_message_type_matches():
    """Notice message should be created with same type as input messages."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),
        Msg("assistant", "a2"),
        Msg("user", "u3"),
        Msg("assistant", "a3"),
        Msg("user", "u4"),
        Msg("assistant", "a4"),
    ]
    result = truncate_conversation(msgs, "half")
    assert len(result.messages) > 3
    notice = result.messages[3]
    assert hasattr(notice, 'role')
    assert notice.role == "user"
    assert "CONTEXT NOTICE" in notice.content


# ---- Edge cases ----

def test_tools_at_start_belong_to_asst():
    """Tools at start of kept_messages belong to the first assistant with tool_calls."""
    # 8 messages, remaining = 5, half = 2 → even = 2
    # kept = [-2:] = [tool(r1), user(u_final)]
    # first_non_tool = 1 (user). user has no tool_calls → tool is orphaned → dropped
    # Expected: system, u1, a1, notice, u_final
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),          # removed
        Msg("assistant", "a2"),      # removed
        Msg("assistant", "caller", tool_calls=[{"id": "1"}]),
        Msg("tool", "result_1", tool_call_id="1"),
        Msg("user", "u_final"),
    ]
    # remaining = [u2, a2, caller, result_1, u_final] = 5
    # half = 2, even = 2
    # kept = [-2:] = [result_1, u_final]
    # first_non_tool = 1 (u_final)
    # u_final has no tool_calls → result_1 is orphaned → dropped
    # kept = [u_final]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert "tool" not in roles, f"Orphaned tool found: {roles}"
    # system + u1 + a1 + notice + u_final = 5
    assert roles == ["system", "user", "assistant", "user", "user"], f"Got: {roles}"


def test_tools_at_start_belong_to_asst_matched():
    """Tool at start + its assistant with tool_calls: both survive."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u_rm1"),
        Msg("assistant", "a_rm1"),
        Msg("user", "u_rm2"),
        Msg("assistant", "a_rm2"),
        Msg("tool", "r1", tool_call_id="1"),
        Msg("tool", "r2", tool_call_id="2"),
        Msg("assistant", "caller", tool_calls=[{"id": "1"}, {"id": "2"}]),
        Msg("user", "u_final"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert "tool" in roles, f"Tools should survive: {roles}"
    assert sum(1 for r in roles if r == "tool") == 2


def test_pull_from_removed_region():
    """Tool results in removed region are pulled when needed by kept assistant."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u_rm1"),
        Msg("assistant", "a_rm1"),
        Msg("user", "u_rm2"),
        Msg("assistant", "a_rm2"),
        Msg("tool", "result_1", tool_call_id="1"),
        Msg("assistant", "caller", tool_calls=[{"id": "1"}]),
        Msg("user", "u_final"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    # result_1 should be pulled back
    assert roles == ["system", "user", "assistant", "user", "tool", "assistant", "user"], \
        f"Got: {roles}"
    assert sum(1 for r in roles if r == "tool") == 1


def test_cannot_pull_enough_from_removed():
    """Not enough pullable tools in removed region → whole group dropped."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u_rm1"),
        Msg("assistant", "a_rm1"),
        Msg("user", "u_rm2"),
        Msg("assistant", "a_rm2"),
        Msg("tool", "result_1", tool_call_id="1"),  # only 1 pullable
        Msg("assistant", "caller", tool_calls=[{"id": "1"}, {"id": "2"}]),  # needs 2
        Msg("user", "u_final"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert "tool" not in roles, f"Orphaned tool should be dropped: {roles}"
    for m in result.messages:
        assert not getattr(m, 'tool_calls', None), f"Orphaned tc should be dropped: {m}"


def test_excess_tools_dropped():
    """When more tools than needed, excess at front are dropped."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u_rm1"),
        Msg("assistant", "a_rm1"),
        Msg("user", "u_rm2"),
        Msg("assistant", "a_rm2"),
        Msg("tool", "extra_result", tool_call_id="extra"),
        Msg("tool", "real_result", tool_call_id="1"),
        Msg("assistant", "caller", tool_calls=[{"id": "1"}]),
        Msg("user", "u_final"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert sum(1 for r in roles if r == "tool") == 1, \
        f"Expected exactly 1 tool, got: {roles}"


def test_no_tool_groups_no_regression():
    """Conversations without tool_calls should be unaffected."""
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),
        Msg("assistant", "a2"),
        Msg("user", "u3"),
        Msg("assistant", "a3"),
        Msg("user", "u4"),
        Msg("assistant", "a4"),
    ]
    result = truncate_conversation(msgs, "half")
    roles = [m.role for m in result.messages]
    assert roles == ["system", "user", "assistant", "user", "user", "assistant"], f"Got: {roles}"


# ---- Bug-fix regression tests ----

def test_first_assistant_with_tool_calls_stripped():
    """First assistant (index 2) has tool_calls whose results are truncated.

    This is a regression test for the compaction bug that caused
    "assistant message with 'tool_calls' must be followed by tool messages"
    errors from providers like DeepSeek.

    Scenario: first assistant (index 2) called tools, results at index 3.
    Compaction cuts the middle, leaving index 2 orphaned.
    """
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1_with_tc", tool_calls=[{"id": "1"}]),  # index 2 has tc
        Msg("tool", "r1", tool_call_id="1"),  # tool result (removed)
        Msg("user", "u2"),
        Msg("assistant", "a2"),
        Msg("user", "u3"),
        Msg("assistant", "a3"),
    ]
    result = truncate_conversation(msgs, "half")
    # The tool result r1 (index 3) falls into the removed region.
    # Index 2 must have its tool_calls stripped to prevent API 400.
    first_asst = result.messages[2]
    assert not _has_tool_calls(first_asst), \
        f"First assistant should have tool_calls stripped: {first_asst}"
    # The tool message should NOT be in the result (it was in removed region)
    roles = [m.role for m in result.messages]
    assert "tool" not in roles, f"Tool message should not survive: {roles}"


def test_first_assistant_tc_dict_style():
    """Same as test_first_assistant_with_tool_calls_stripped but with dict messages."""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    result = truncate_conversation(msgs, "half")
    first_asst = result.messages[2]
    tc = first_asst.get("tool_calls") if isinstance(first_asst, dict) else getattr(first_asst, "tool_calls", None)
    assert not tc, f"First assistant should have tool_calls stripped: {first_asst}"


def test_second_tc_group_orphaned_after_first():
    """Second tool_calls group in kept_messages has missing tool results.

    The original truncation algorithm only validates the first group.
    This tests that the safety net catches subsequent orphaned groups.
    """
    # 10 messages, half => remaining=7, keep=2
    # kept = [user_kept, asst2(tc=1)] where asst2's tool result was truncated
    msgs = [
        Msg("system", "s"),
        Msg("user", "u1"),
        Msg("assistant", "a1"),
        Msg("user", "u2"),          # removed
        Msg("assistant", "a2"),      # removed
        Msg("assistant", "asst1", tool_calls=[{"id": "1"}]),  # removed
        Msg("tool", "r1", tool_call_id="1"),                   # removed
        Msg("user", "u_kept"),       # kept
        Msg("assistant", "asst2", tool_calls=[{"id": "3"}]),  # kept, orphaned
        # asst2's tool result was at index 9 -- truncated
    ]
    result = truncate_conversation(msgs, "half")
    # asst2 should have tool_calls stripped
    for m in result.messages:
        role = m.role if hasattr(m, 'role') else m.get('role', '')
        tc = getattr(m, 'tool_calls', None) or (m.get('tool_calls') if isinstance(m, dict) else None)
        if role == "assistant" and tc:
            for t in tc:
                t_id = t.get("id", "") if isinstance(t, dict) else (getattr(t, "id", "") if hasattr(t, "id") else "")
                if t_id == "3":
                    assert False, f"asst2 should have tool_calls stripped: {m}"
