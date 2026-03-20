"""Tests for tool call parser recovery from malformed model output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harness.cline_agent import (
    parse_xml_tool,
    _normalize_tool_xml,
    strip_thinking_blocks,
)


def test_hybrid_tool_call_execute():
    """Model outputs <tool_call>execute_command> instead of <execute_command>."""
    content = (
        "<thinking>Some reasoning</thinking>"
        "<tool_call>execute_command>\n"
        '<command>dir "C:\\projects\\evoke"</command>\n'
        "</execute_command>\n"
        "</thinking>\n"
    )
    result = parse_xml_tool(content)
    assert result is not None
    assert result.name == "execute_command"
    assert result.parameters["command"] == 'dir "C:\\projects\\evoke"'


def test_hybrid_tool_call_read_file():
    """Model outputs <tool_call>read_file> instead of <read_file>."""
    content = (
        "<thinking>Good</thinking>"
        "<tool_call>read_file>\n"
        "<path>examples/llm_smollm2/main.go</path>\n"
        "</read_file>\n"
        "</thinking>\n"
    )
    result = parse_xml_tool(content)
    assert result is not None
    assert result.name == "read_file"
    assert result.parameters["path"] == "examples/llm_smollm2/main.go"


def test_normal_format_still_works():
    """Standard tool call format is unaffected."""
    content = "<execute_command>\n<command>git pull</command>\n</execute_command>"
    result = parse_xml_tool(content)
    assert result is not None
    assert result.name == "execute_command"
    assert result.parameters["command"] == "git pull"


def test_normalize_tool_xml():
    """_normalize_tool_xml converts <tool_call>name> to <name>."""
    raw = "<tool_call>execute_command>\n<command>test</command>\n</execute_command>"
    normalized = _normalize_tool_xml(raw)
    assert "<execute_command>" in normalized
    assert "<tool_call>" not in normalized


def test_strip_thinking_orphaned_close():
    """strip_thinking_blocks removes orphaned </thinking> tags."""
    content = "<thinking>foo</thinking>bar</thinking>"
    stripped = strip_thinking_blocks(content)
    assert "</thinking>" not in stripped
    assert "bar" in stripped


def test_thinking_only_no_tool():
    """Pure thinking block with no tool returns None."""
    content = "<thinking>The user wants me to continue.</thinking>\n"
    result = parse_xml_tool(content)
    assert result is None


def test_hybrid_with_manage_todos():
    """Model wraps manage_todos in <tool_call>."""
    content = (
        "<thinking>Need to add todo</thinking>\n"
        "<tool_call>manage_todos>\n"
        "<action>add</action>\n"
        "<title>Fix auth bug</title>\n"
        "</manage_todos>"
    )
    result = parse_xml_tool(content)
    assert result is not None
    assert result.name == "manage_todos"
    assert result.parameters["action"] == "add"
    assert result.parameters["title"] == "Fix auth bug"


def test_tool_call_inside_thinking_block():
    """Tool call wrapped entirely inside thinking — should still parse after stripping."""
    content = (
        "<thinking>\n"
        "Need to check.\n"
        "</thinking>\n"
        "<tool_call>execute_command>\n"
        "<command>ls</command>\n"
        "</execute_command>\n"
        "</thinking>\n"
    )
    result = parse_xml_tool(content)
    assert result is not None
    assert result.name == "execute_command"
    assert result.parameters["command"] == "ls"
