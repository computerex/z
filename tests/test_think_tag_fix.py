#!/usr/bin/env python3
"""Tests for <think>/<thinking> tag handling in the harness.

Verifies:
1. strip_thinking_blocks properly strips both <thinking> and <think> blocks
2. strip_thinking_blocks preserves orphaned </think> (user content)
3. XML stream filter properly routes <think> blocks to reasoning display
4. XML stream filter displays orphaned </think> to the terminal
5. Tool parsing works correctly when model output includes <think> reasoning
6. User input containing </think> survives the message serialization pipeline
"""

import re
import sys
import json
import io
from typing import Dict, List, Optional

# Import from the harness source
sys.path.insert(0, "src")
from harness.cline_agent import (
    strip_thinking_blocks,
    parse_xml_tool,
    parse_all_xml_tools,
)
from harness.streaming_client import StreamingMessage


# ─── strip_thinking_blocks ───────────────────────────────────────────────


def test_strip_thinking_blocks_removes_thinking():
    """<thinking>...</thinking> blocks are fully removed."""
    content = "<thinking>deep reasoning here</thinking>\nHello world"
    result = strip_thinking_blocks(content)
    assert "<thinking>" not in result
    assert "deep reasoning" not in result
    assert "Hello world" in result


def test_strip_thinking_blocks_removes_think():
    """<think>...</think> blocks (used by DeepSeek/Qwen etc.) are fully removed."""
    content = "<think>model reasoning about the task</think>\nHere is my response"
    result = strip_thinking_blocks(content)
    assert "<think>" not in result
    assert "model reasoning" not in result
    assert "Here is my response" in result


def test_strip_thinking_blocks_removes_orphaned_closing_thinking():
    """Orphaned </thinking> tags are removed."""
    content = "Some text</thinking> more text"
    result = strip_thinking_blocks(content)
    assert "</thinking>" not in result
    assert "Some text" in result
    assert "more text" in result


def test_strip_thinking_blocks_preserves_orphaned_closing_think():
    """Orphaned </think> tags are PRESERVED — they may be user content."""
    content = "Here is what you said: </think>"
    result = strip_thinking_blocks(content)
    assert "</think>" in result
    assert "Here is what you said: </think>" == result


def test_strip_thinking_blocks_model_reasoning_plus_user_think():
    """<think> reasoning is stripped, but orphaned </think> in response is preserved."""
    content = (
        "<think>The user wants me to repeat the closing think tag</think>\n"
        "Here is what you said: </think>"
    )
    result = strip_thinking_blocks(content)
    assert "<think>" not in result
    assert "user wants me to repeat" not in result
    assert "</think>" in result
    assert "Here is what you said: </think>" in result.strip()


def test_strip_thinking_blocks_both_variants():
    """Both <thinking> and <think> blocks in the same content."""
    content = (
        "<thinking>first pass</thinking>"
        "<think>second pass</think>"
        "actual response"
    )
    result = strip_thinking_blocks(content)
    assert "first pass" not in result
    assert "second pass" not in result
    assert "actual response" in result


def test_strip_thinking_blocks_multiline_think():
    """Multiline <think> blocks with newlines."""
    content = (
        "<think>\nLine 1\nLine 2\nLine 3\n</think>\n"
        "The answer is 42"
    )
    result = strip_thinking_blocks(content)
    assert "<think>" not in result
    assert "Line 1" not in result
    assert "The answer is 42" in result


def test_strip_thinking_blocks_empty_content():
    """Empty and whitespace-only content."""
    assert strip_thinking_blocks("") == ""
    assert strip_thinking_blocks("   ") == "   "


def test_strip_thinking_blocks_no_tags():
    """Content without any think/thinking tags passes through unchanged."""
    content = "Hello, I'm a regular response with no special tags."
    assert strip_thinking_blocks(content) == content


# ─── Tool parsing with <think> blocks ────────────────────────────────────


def test_parse_tool_with_think_reasoning():
    """Tool calls parse correctly even when preceded by <think> reasoning."""
    content = (
        "<think>Let me read the file to check</think>\n"
        "<read_file>\n<path>test.py</path>\n</read_file>"
    )
    tool = parse_xml_tool(content)
    assert tool is not None
    assert tool.name == "read_file"
    assert tool.parameters.get("path") == "test.py"


def test_parse_tool_with_think_containing_tool_like_text():
    """<think> block with tool-like text inside doesn't confuse parser."""
    content = (
        "<think>Should I use read_file? Let me check the path first.</think>\n"
        "<execute_command>\n<command>ls -la</command>\n</execute_command>"
    )
    tool = parse_xml_tool(content)
    assert tool is not None
    assert tool.name == "execute_command"
    assert "ls -la" in tool.parameters.get("command", "")


def test_parse_all_tools_with_think_blocks():
    """parse_all_xml_tools correctly strips <think> before parsing."""
    content = (
        "<think>Planning my approach</think>\n"
        "<read_file>\n<path>a.py</path>\n</read_file>\n"
        "<read_file>\n<path>b.py</path>\n</read_file>"
    )
    tools = parse_all_xml_tools(content)
    assert len(tools) == 2
    assert tools[0].parameters.get("path") == "a.py"
    assert tools[1].parameters.get("path") == "b.py"


# ─── XML stream filter simulation ────────────────────────────────────────


def _simulate_stream_filter(text: str) -> tuple:
    """Simulate the _sf_char XML stream filter.

    Returns (displayed_text, reasoning_text) — what the user sees
    on the terminal vs. what goes to the reasoning display.
    """
    from harness.tool_registry import get_tool_names

    _sf_tool_names = set(get_tool_names()) | {"tool_call"}
    _sf_thinking_tags = {"thinking", "think"}
    _sf_suppressing: Optional[str] = None
    _sf_thinking_suppress = False
    _sf_tag_buf = ""
    _sf_in_tag = False

    displayed = io.StringIO()
    reasoning = io.StringIO()

    def _sf_flush():
        nonlocal _sf_tag_buf
        if _sf_tag_buf:
            displayed.write(_sf_tag_buf)
            _sf_tag_buf = ""

    def on_reasoning(text):
        reasoning.write(text)

    for c in text:
        # SUPPRESS_BLOCK
        if _sf_suppressing:
            _sf_tag_buf += c
            close = f"</{_sf_suppressing}>"
            if _sf_tag_buf.endswith(close):
                _sf_suppressing = None
                _sf_tag_buf = ""
            elif len(_sf_tag_buf) > len(close) + 30:
                _sf_tag_buf = _sf_tag_buf[-(len(close) + 10):]
            continue

        # THINKING_SUPPRESS
        if _sf_thinking_suppress:
            _sf_tag_buf += c
            for tag in _sf_thinking_tags:
                close = f"</{tag}>"
                if _sf_tag_buf.endswith(close):
                    t = _sf_tag_buf[:-len(close)]
                    if t:
                        on_reasoning(t)
                    _sf_thinking_suppress = False
                    _sf_tag_buf = ""
                    break
            else:
                if len(_sf_tag_buf) > 200:
                    on_reasoning(_sf_tag_buf[:-20])
                    _sf_tag_buf = _sf_tag_buf[-20:]
            continue

        # DETECT_TAG
        if _sf_in_tag:
            _sf_tag_buf += c
            if c == ">":
                m = re.match(r"</?(\w+)", _sf_tag_buf)
                if m:
                    tag = m.group(1)
                    is_close = _sf_tag_buf.startswith("</")
                    if tag in _sf_tool_names:
                        if not is_close:
                            _sf_suppressing = tag
                        _sf_tag_buf = ""
                        _sf_in_tag = False
                        continue
                    elif tag in _sf_thinking_tags:
                        if is_close:
                            # Orphaned close tag — display as-is
                            _sf_flush()
                        else:
                            _sf_thinking_suppress = True
                            _sf_tag_buf = ""
                        _sf_in_tag = False
                        continue
                _sf_flush()
                _sf_in_tag = False
                continue
            if len(_sf_tag_buf) > 80:
                _sf_flush()
                _sf_in_tag = False
            continue

        # NORMAL
        if c == "<":
            _sf_in_tag = True
            _sf_tag_buf = c
            continue

        displayed.write(c)

    # Flush remaining
    _sf_flush()

    return displayed.getvalue(), reasoning.getvalue()


def test_filter_think_block_routes_to_reasoning():
    """<think>...</think> block content is routed to reasoning display."""
    text = "<think>deep analysis here</think>\nVisible response"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "deep analysis here" in reasoning
    assert "deep analysis here" not in displayed
    assert "Visible response" in displayed


def test_filter_thinking_block_routes_to_reasoning():
    """<thinking>...</thinking> block content is routed to reasoning display."""
    text = "<thinking>analysis</thinking>\nResponse"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "analysis" in reasoning
    assert "analysis" not in displayed
    assert "Response" in displayed


def test_filter_orphaned_close_think_displayed():
    """Orphaned </think> (not inside a thinking block) is displayed to user."""
    text = "Here is what you said: </think>"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "</think>" in displayed
    assert "Here is what you said: </think>" == displayed


def test_filter_orphaned_close_thinking_displayed():
    """Orphaned </thinking> is displayed (not silently eaten)."""
    text = "Response with </thinking> in it"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "</thinking>" in displayed


def test_filter_think_block_then_orphaned_close():
    """<think> block is suppressed, then orphaned </think> in content is displayed."""
    text = "<think>reasoning stuff</think>\nHere: </think>"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "reasoning stuff" in reasoning
    assert "reasoning stuff" not in displayed
    assert "</think>" in displayed
    assert "Here: </think>" in displayed.strip()


def test_filter_tool_tags_suppressed():
    """Tool tags like <read_file> are still suppressed (regression check)."""
    text = "Let me check <read_file><path>test.py</path></read_file> done"
    displayed, reasoning = _simulate_stream_filter(text)
    assert "<read_file>" not in displayed
    assert "<path>" not in displayed
    assert "test.py" not in displayed
    assert "Let me check" in displayed
    assert "done" in displayed


def test_filter_plain_text_unchanged():
    """Plain text without any special tags passes through unchanged."""
    text = "Hello world, no tags here!"
    displayed, reasoning = _simulate_stream_filter(text)
    assert displayed == text
    assert reasoning == ""


# ─── Empty response detection ────────────────────────────────────────────


def test_short_response_not_treated_as_empty():
    """Short but valid responses must NOT trigger the empty-response nudge.

    The old code used `len(display_text) < 20` which falsely caught
    legitimate short answers like 'repeat this', 'yes', '42', etc.
    """
    # These are all valid model responses that should NOT be "empty"
    short_responses = [
        "<think>reasoning about it</think>\nrepeat this",
        "<think>let me think</think>\nyes",
        "<think>calculating</think>\n42",
        "ok",
        "</think>",
        "hi",
        "<think>deep thought</think>\n!",
    ]
    for resp in short_responses:
        stripped = strip_thinking_blocks(resp).strip()
        # The key invariant: if stripped text exists, it's not empty
        assert len(stripped) > 0 or resp in [""], (
            f"Response {resp!r} stripped to empty: {stripped!r}"
        )


def test_truly_empty_response_detected():
    """A response that is ONLY thinking with nothing after is truly empty."""
    empty_responses = [
        "<think>just thinking, no response</think>",
        "<thinking>analysis only</thinking>",
        "<think>hmm</think>\n",
        "<thinking>deep thought</thinking>\n  \n",
    ]
    for resp in empty_responses:
        stripped = strip_thinking_blocks(resp).strip()
        assert stripped == "", (
            f"Response {resp!r} should be empty after stripping, got: {stripped!r}"
        )


# ─── User input pipeline ─────────────────────────────────────────────────


def test_user_input_think_preserved_in_message():
    """</think> in user input is preserved in StreamingMessage."""
    msg = StreamingMessage(role="user", content="repeat this: </think>")
    assert "</think>" in msg.content
    d = msg.to_dict()
    assert "</think>" in d["content"]


def test_user_input_think_preserved_in_json():
    """</think> in user message survives JSON serialization."""
    msg = StreamingMessage(role="user", content="</think>")
    payload = json.dumps([msg.to_dict()], ensure_ascii=False)
    assert "</think>" in payload
    roundtrip = json.loads(payload)
    assert "</think>" in roundtrip[0]["content"]


def test_user_input_think_in_conversation():
    """Full conversation with </think> in user message."""
    messages = [
        StreamingMessage(role="system", content="You are a helpful assistant."),
        StreamingMessage(role="user", content="hi repeat after me ok?"),
        StreamingMessage(role="assistant", content="OK, go ahead!"),
        StreamingMessage(role="user", content="</think>"),
    ]
    payload = json.dumps([m.to_dict() for m in messages], ensure_ascii=False)
    data = json.loads(payload)
    last_user = [m for m in data if m["role"] == "user"][-1]
    assert last_user["content"] == "</think>"


# ─── Full pipeline simulation ────────────────────────────────────────────


def test_full_pipeline_model_repeats_think_tag():
    """End-to-end: model response repeating </think> is displayed and preserved."""
    # Simulate model output: reasoning in <think> block, then response with </think>
    model_output = (
        "<think>The user wants me to output the literal text '</think>'."
        " I should include it in my response.</think>\n"
        "Here is what you asked me to repeat:\n\n</think>"
    )

    # 1. Display filter shows the response part with </think>
    displayed, reasoning = _simulate_stream_filter(model_output)
    assert "</think>" in displayed, f"</think> missing from display: {displayed!r}"
    assert "Here is what you asked me to repeat" in displayed

    # 2. strip_thinking_blocks preserves orphaned </think>
    stripped = strip_thinking_blocks(model_output)
    assert "</think>" in stripped
    assert "<think>" not in stripped
    assert "user wants me to output" not in stripped
    assert "Here is what you asked me to repeat" in stripped


def test_full_pipeline_no_reasoning_just_think_tag():
    """Model without reasoning simply outputs </think> — nothing stripped."""
    model_output = "Sure! Here is what you said: </think>"

    displayed, reasoning = _simulate_stream_filter(model_output)
    assert displayed == model_output
    assert reasoning == ""

    stripped = strip_thinking_blocks(model_output)
    assert stripped == model_output


# ─── ZWS sanitization / desanitization (transport-layer escaping) ────────

from harness.streaming_client import (
    _sanitize_think_tokens,
    _sanitize_messages_for_api,
    desanitize_think_tokens,
    _THINK_ZWS,
)


def test_sanitize_think_tokens_basic():
    """<think> and </think> are escaped with ZWS; <thinking> is NOT."""
    text = "before <think>reasoning</think> after <thinking>ok</thinking>"
    result = _sanitize_think_tokens(text)
    assert f"<{_THINK_ZWS}think>" in result
    assert f"<{_THINK_ZWS}/think>" in result
    assert "<thinking>" in result  # NOT escaped
    assert "</thinking>" in result  # NOT escaped
    assert "<think>" not in result  # original form gone
    assert "</think>" not in result


def test_sanitize_does_not_touch_thinking():
    """Ensure <thinking> tags pass through unmodified."""
    text = "<thinking>deep thought</thinking>"
    assert _sanitize_think_tokens(text) == text


def test_desanitize_round_trip():
    """sanitize → desanitize gives back the original text."""
    original = "file has </think> and <think> tokens"
    assert desanitize_think_tokens(_sanitize_think_tokens(original)) == original


def test_sanitize_messages_for_api_str_content():
    """String-content messages are sanitized."""
    msgs = [{"role": "user", "content": "see </think> here"}]
    result = _sanitize_messages_for_api(msgs)
    assert "</think>" not in result[0]["content"]
    assert f"<{_THINK_ZWS}/think>" in result[0]["content"]


def test_sanitize_messages_for_api_list_content():
    """List-content messages (multimodal) are sanitized."""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "look: </think>"},
        {"type": "image_url", "url": "http://x.png"},
    ]}]
    result = _sanitize_messages_for_api(msgs)
    assert "</think>" not in result[0]["content"][0]["text"]
    assert result[0]["content"][1] == {"type": "image_url", "url": "http://x.png"}


def test_sanitize_messages_no_mutation():
    """Original messages_dict is not mutated."""
    msgs = [{"role": "user", "content": "has </think>"}]
    original_content = msgs[0]["content"]
    _sanitize_messages_for_api(msgs)
    assert msgs[0]["content"] == original_content  # unchanged


def test_desanitize_in_tool_params():
    """Tool handler desanitization reverses ZWS for replace_in_file params."""
    # Simulate: model copies ZWS-escaped token from read_file result
    escaped = f"<{_THINK_ZWS}/think>"
    assert desanitize_think_tokens(escaped) == "</think>"

    # Simulate: model outputs raw </think> (no ZWS) — should pass through
    assert desanitize_think_tokens("</think>") == "</think>"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
