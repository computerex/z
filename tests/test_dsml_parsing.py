"""Tests for DSML (DeepSeek Markup Language) tool call parsing."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from harness.context_management import (
    parse_dsml_tool_calls,
    strip_dsml_tags,
)


def test_no_dsml_noop():
    """Content without DSML tags is returned unchanged."""
    tc, cleaned = parse_dsml_tool_calls("Hello world")
    assert tc == []
    assert cleaned == "Hello world"


def test_single_tool_call():
    """A single DSML tool call is parsed correctly."""
    content = (
        'Let me read the file.\n'
        '<tool_calls>\n'
        '<invoke name="read_file">\n'
        '<parameter name="path">/etc/hosts</parameter>\n'
        '</invoke>\n'
        '</tool_calls>'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 1, f"Expected 1 tool call, got {len(tc)}: {tc}"
    assert tc[0]["function"]["name"] == "read_file"
    args = json.loads(tc[0]["function"]["arguments"])
    assert args == {"path": "/etc/hosts"}
    assert cleaned == "Let me read the file."
    assert "tool_calls" not in cleaned
    assert "invoke" not in cleaned
    assert "parameter" not in cleaned


def test_multiple_tool_calls():
    """Multiple DSML tool calls are parsed correctly."""
    content = (
        '<tool_calls>\n'
        '<invoke name="execute_command">\n'
        '<parameter name="command">ls -la</parameter>\n'
        '</invoke>\n'
        '<invoke name="read_file">\n'
        '<parameter name="path">/etc/hosts</parameter>\n'
        '</invoke>\n'
        '</tool_calls>\n'
        'Done.'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}: {tc}"
    assert tc[0]["function"]["name"] == "execute_command"
    assert tc[1]["function"]["name"] == "read_file"
    assert cleaned == "Done."
    assert "<tool_calls>" not in cleaned


def test_dsml_with_text_before_after():
    """Text before and after DSML block is preserved, tags stripped."""
    content = (
        'I will check the file.\n'
        '<tool_calls>\n'
        '<invoke name="read_file">\n'
        '<parameter name="path">config.json</parameter>\n'
        '</invoke>\n'
        '</tool_calls>\n'
        'Here is what I found.'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 1
    # Newlines from removed tags are collapsed to at most 2
    assert cleaned == "I will check the file.\n\nHere is what I found.", repr(cleaned)


def test_multiple_parameters():
    """Tool calls with multiple parameters."""
    content = (
        '<tool_calls>\n'
        '<invoke name="search_files">\n'
        '<parameter name="regex">foo</parameter>\n'
        '<parameter name="path">/src</parameter>\n'
        '</invoke>\n'
        '</tool_calls>'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "search_files"
    args = json.loads(tc[0]["function"]["arguments"])
    assert args == {"regex": "foo", "path": "/src"}


def test_no_tool_calls_tags():
    """<invoke> without <tool_calls> wrapper is NOT parsed."""
    content = (
        '<invoke name="read_file">\n'
        '<parameter name="path">/etc/hosts</parameter>\n'
        '</invoke>'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert tc == [], f"Expected no tool calls without <tool_calls> wrapper: {tc}"
    assert cleaned == content


def test_strip_only():
    """strip_dsml_tags removes DSML tags without parsing."""
    content = (
        '<tool_calls>\n'
        '<invoke name="execute_command">\n'
        '<parameter name="command">ls</parameter>\n'
        '</invoke>\n'
        '</tool_calls>'
    )
    cleaned = strip_dsml_tags(content)
    assert cleaned == ""
    assert "<tool_calls>" not in cleaned
    assert "<invoke" not in cleaned
    assert "<parameter" not in cleaned


def test_strip_preserves_non_dsml_text():
    """strip_dsml_tags preserves surrounding text."""
    content = (
        'Before.\n'
        '<tool_calls>\n'
        '<invoke name="foo"><parameter name="x">1</parameter></invoke>\n'
        '</tool_calls>\n'
        'After.'
    )
    cleaned = strip_dsml_tags(content)
    # Newlines from removed tags are collapsed
    assert cleaned == "Before.\n\nAfter.", repr(cleaned)


def test_dsml_inline_format():
    """DSML tags on a single line (no newlines) are handled."""
    content = (
        'Let me run it. '
        '<tool_calls><invoke name="execute_command">'
        '<parameter name="command">./binary</parameter>'
        '</invoke></tool_calls>'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "execute_command"
    args = json.loads(tc[0]["function"]["arguments"])
    assert args == {"command": "./binary"}
    assert cleaned == "Let me run it."


def test_empty_parameters():
    """Empty parameter values are handled."""
    content = (
        '<tool_calls>\n'
        '<invoke name="foo">\n'
        '<parameter name="x"></parameter>\n'
        '</invoke>\n'
        '</tool_calls>'
    )
    tc, cleaned = parse_dsml_tool_calls(content)
    assert len(tc) == 1
    args = json.loads(tc[0]["function"]["arguments"])
    assert args == {"x": ""}

def test_stray_closing_tag_stripped():
    """A stray </tool_calls> without opening tag is stripped by strip_dsml_tags."""
    content = "Let me run it. </tool_calls>"
    cleaned = strip_dsml_tags(content)
    assert cleaned == "Let me run it.", repr(cleaned)
    # parse_dsml_tool_calls should NOT detect this as a tool call
    tc, _ = parse_dsml_tool_calls(content)
    assert tc == [], f"Stray closing tag should not produce tools: {tc}"


def test_stray_invoke_stripped():
    """Stray <invoke> without <tool_calls> wrapper is stripped."""
    content = '<invoke name="foo"><parameter name="x">1</parameter></invoke>'
    cleaned = strip_dsml_tags(content)
    assert cleaned == "", repr(cleaned)


def test_strip_independent_of_parse():
    """strip_dsml_tags works on content that parse_dsml_tool_calls ignores."""
    # Content with stray closing tags but no opening <tool_calls>
    content = "Some text </tool_calls> more text"
    # parse ignores it (no <tool_calls> wrapper)
    tc, parse_cleaned = parse_dsml_tool_calls(content)
    assert tc == []
    assert parse_cleaned == content  # parse returns content unchanged
    # strip actively removes the stray tag
    strip_cleaned = strip_dsml_tags(content)
    assert strip_cleaned == "Some text  more text", repr(strip_cleaned)

