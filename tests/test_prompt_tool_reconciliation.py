"""Reconciliation tests: verify the system prompt accurately documents
every tool defined in the tool registry, and every tool has a handler.

These tests catch the class of bug where a tool handler supports parameters
(like read_file's start_line/end_line) but the system prompt never tells
the model about them — so the model can't use them.
"""

import re
import sys
from pathlib import Path

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.tool_registry import TOOL_DEFS, TOOL_BY_NAME
from harness.prompts import get_system_prompt

# Generate a concrete system prompt to parse
_PROMPT = get_system_prompt(workspace_path="/test/workspace")


def _extract_prompt_tool_sections() -> dict[str, str]:
    """Parse '## tool_name' sections out of the system prompt.

    Returns {tool_name: full_section_text} for every tool documented in
    the prompt.
    """
    sections: dict[str, str] = {}
    # Split on markdown H2 headers
    parts = re.split(r"(?m)^## ", _PROMPT)
    for part in parts[1:]:  # skip everything before first ##
        lines = part.split("\n", 1)
        name = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        sections[name] = body
    return sections


def _extract_prompt_params(section_text: str) -> dict[str, bool]:
    """Extract parameter names and required-ness from a prompt section.

    Handles multiple formats used in the prompt:
      - param: (required) ...
      - param: (optional) ...
      - param: Description text (no explicit marker → optional)
    Returns {param_name: is_required}.
    """
    params: dict[str, bool] = {}
    for m in re.finditer(r"^- (\w+):", section_text, re.MULTILINE):
        name = m.group(1)
        rest_of_line = section_text[m.end():].split("\n", 1)[0]
        is_required = "(required)" in rest_of_line
        params[name] = is_required
    return params


# ── Registry → Prompt reconciliation ──────────────────────────────

# Tools handled directly in the agent loop, not via tool_handlers
_AGENT_LOOP_TOOLS = {"attempt_completion", "manage_todos", "introspect"}


class TestRegistryToPrompt:
    """Every tool in TOOL_DEFS must have a matching ## section in the prompt."""

    def test_every_registry_tool_documented_in_prompt(self):
        prompt_sections = _extract_prompt_tool_sections()
        missing = []
        for td in TOOL_DEFS:
            if td.name not in prompt_sections:
                missing.append(td.name)
        assert not missing, (
            f"Tools in registry but missing from system prompt: {missing}\n"
            f"Add a '## {missing[0]}' section to prompts.py"
        )

    def test_every_prompt_tool_exists_in_registry(self):
        prompt_sections = _extract_prompt_tool_sections()
        extra = []
        for name in prompt_sections:
            if name not in TOOL_BY_NAME:
                extra.append(name)
        assert not extra, (
            f"Tools documented in prompt but missing from registry: {extra}"
        )


class TestParamReconciliation:
    """Every param in the registry must be documented in the prompt, and vice versa."""

    def test_every_registry_param_documented_in_prompt(self):
        prompt_sections = _extract_prompt_tool_sections()
        missing = []
        for td in TOOL_DEFS:
            section = prompt_sections.get(td.name, "")
            prompt_params = _extract_prompt_params(section)
            for p in td.params:
                if p.name not in prompt_params:
                    missing.append(f"{td.name}.{p.name}")
        assert not missing, (
            f"Params defined in registry but undocumented in prompt: {missing}\n"
            f"The model cannot use parameters it doesn't know about."
        )

    def test_required_params_match(self):
        """Required-ness in registry matches required-ness in prompt."""
        prompt_sections = _extract_prompt_tool_sections()
        mismatches = []
        for td in TOOL_DEFS:
            section = prompt_sections.get(td.name, "")
            prompt_params = _extract_prompt_params(section)
            for p in td.params:
                if p.name in prompt_params:
                    if p.required != prompt_params[p.name]:
                        expected = "required" if p.required else "optional"
                        actual = "required" if prompt_params[p.name] else "optional"
                        mismatches.append(
                            f"{td.name}.{p.name}: registry={expected}, prompt={actual}"
                        )
        assert not mismatches, (
            f"Required-ness mismatch between registry and prompt:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )

    def test_no_extra_prompt_params(self):
        """Prompt doesn't document params that don't exist in the registry."""
        prompt_sections = _extract_prompt_tool_sections()
        extras = []
        for td in TOOL_DEFS:
            section = prompt_sections.get(td.name, "")
            prompt_params = _extract_prompt_params(section)
            registry_param_names = {p.name for p in td.params}
            for pp_name in prompt_params:
                if pp_name not in registry_param_names:
                    extras.append(f"{td.name}.{pp_name}")
        assert not extras, (
            f"Params documented in prompt but missing from registry: {extras}\n"
            f"Parser won't extract these params correctly."
        )


class TestUsageExamplesMatchParams:
    """The XML usage examples in the prompt should use the tool's actual params."""

    def test_usage_xml_tags_match_registry_params(self):
        prompt_sections = _extract_prompt_tool_sections()
        issues = []
        for td in TOOL_DEFS:
            section = prompt_sections.get(td.name, "")
            # Find all XML-like tags in usage examples: <param_name>...</param_name>
            usage_params = set(re.findall(
                rf"<({td.name})>.*?</\1>", section, re.DOTALL
            ))
            # Find inner param tags
            inner_tags = set()
            for block_match in re.finditer(
                rf"<{re.escape(td.name)}>(.*?)</{re.escape(td.name)}>",
                section, re.DOTALL
            ):
                block = block_match.group(1)
                inner_tags.update(re.findall(r"<(\w+)>", block))

            registry_param_names = {p.name for p in td.params}
            # Every tag used in usage should be a real param
            for tag in inner_tags:
                if tag not in registry_param_names:
                    issues.append(
                        f"{td.name}: usage example has <{tag}> but "
                        f"registry params are {registry_param_names}"
                    )
        assert not issues, (
            "Usage examples reference non-existent params:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )


class TestToolDispatchCoverage:
    """Every tool in the registry must be dispatchable (has a handler)."""

    def test_every_tool_has_dispatch_handler(self):
        """Check that _dispatch_tool in cline_agent.py handles every registry tool."""
        agent_path = Path(__file__).resolve().parent.parent / "src" / "harness" / "cline_agent.py"
        source = agent_path.read_text(encoding="utf-8")

        # Find all tool names referenced in _dispatch_tool elif/if chain
        dispatch_re = re.compile(
            r'tool\.name\s*==\s*["\'](\w+)["\']'
        )
        dispatched = set(dispatch_re.findall(source))

        missing = []
        for td in TOOL_DEFS:
            # attempt_completion is handled outside _dispatch_tool
            if td.name == "attempt_completion":
                continue
            if td.name not in dispatched:
                missing.append(td.name)
        assert not missing, (
            f"Tools in registry with no dispatch handler in cline_agent.py: {missing}"
        )

    def test_every_tool_handler_method_exists(self):
        """Check that ToolHandlers class has methods for handler-based tools."""
        from harness.tool_handlers import ToolHandlers

        # Map from tool_name to expected method name on ToolHandlers
        # Tools handled in the agent loop directly are excluded.
        _METHOD_MAP = {
            "read_file": "read_file",
            "write_to_file": "write_file",
            "replace_in_file": "replace_in_file",
            "replace_between_anchors": "replace_between_anchors",
            "execute_command": "execute_command",
            "list_files": "list_files",
            "search_files": "search_files",
            "check_background_process": "check_background_process",
            "stop_background_process": "stop_background_process",
            "list_background_processes": "list_background_processes",
            "analyze_image": "analyze_image",
            "web_search": "web_search",
            "mcp_list_tools": "mcp_list_tools",
            "mcp_search_tools": "mcp_search_tools",
            "mcp_call_tool": "mcp_call_tool",
            "retrieve_tool_result": "retrieve_tool_result",
        }
        missing = []
        for tool_name, method_name in _METHOD_MAP.items():
            if not hasattr(ToolHandlers, method_name):
                missing.append(f"{tool_name} → ToolHandlers.{method_name}()")
        assert not missing, (
            f"Tool handler methods missing: {missing}"
        )


class TestPromptToolParamAcceptance:
    """Verify the tool handler actually reads the params the prompt documents."""

    def test_read_file_handler_uses_start_end_line(self):
        """The exact bug we missed: read_file supports start_line/end_line
        but the prompt didn't document them. Verify the handler reads them."""
        import inspect
        from harness.tool_handlers import ToolHandlers

        source = inspect.getsource(ToolHandlers.read_file)
        assert "start_line" in source, "read_file handler doesn't reference start_line"
        assert "end_line" in source, "read_file handler doesn't reference end_line"

    def test_execute_command_handler_uses_background(self):
        """execute_command has an optional 'background' param."""
        import inspect
        from harness.tool_handlers import ToolHandlers

        source = inspect.getsource(ToolHandlers.execute_command)
        assert "background" in source, "execute_command handler doesn't reference background"

    def test_search_files_handler_uses_file_pattern(self):
        """search_files has an optional 'file_pattern' param."""
        import inspect
        from harness.tool_handlers import ToolHandlers

        source = inspect.getsource(ToolHandlers.search_files)
        assert "file_pattern" in source, "search_files handler doesn't reference file_pattern"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
