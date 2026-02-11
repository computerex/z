"""Test script for SEARCH/REPLACE block parsing and execution.

This tests that our harness can reliably parse and apply SEARCH/REPLACE blocks,
especially when content contains XML-like tags that could confuse the parser.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harness.tool_handlers import parse_search_replace_blocks
from harness.cline_agent import parse_xml_tool


def test_basic_search_replace():
    """Test basic SEARCH/REPLACE block parsing."""
    diff = '''
<<<<<<< SEARCH
func hello() {
    return "hello"
}
=======
func hello() {
    return "hello world"
}
>>>>>>> REPLACE
'''
    blocks = parse_search_replace_blocks(diff)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    search, replace = blocks[0]
    assert "hello" in search
    assert "hello world" in replace
    print("✓ test_basic_search_replace passed")


def test_multiple_blocks():
    """Test multiple SEARCH/REPLACE blocks."""
    diff = '''
<<<<<<< SEARCH
line1
=======
newline1
>>>>>>> REPLACE

<<<<<<< SEARCH
line2
=======
newline2
>>>>>>> REPLACE
'''
    blocks = parse_search_replace_blocks(diff)
    assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"
    print("✓ test_multiple_blocks passed")


def test_xml_content_in_blocks():
    """Test SEARCH/REPLACE blocks containing XML-like content."""
    diff = '''
<<<<<<< SEARCH
## read_file

<read_file>
<path>example.py</path>
</read_file>
=======
## read_file

Use this tool to read files:

<read_file>
<path>path/to/file</path>
</read_file>
>>>>>>> REPLACE
'''
    blocks = parse_search_replace_blocks(diff)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    search, replace = blocks[0]
    assert "<read_file>" in search
    assert "<read_file>" in replace
    assert "path/to/file" in replace
    print("✓ test_xml_content_in_blocks passed")


def test_parse_xml_tool_last_match():
    """Test that parse_xml_tool returns the LAST match, not the first."""
    # Simulate a response that has examples in documentation followed by actual tool call
    content = '''
Here's how to use read_file:

<read_file>
<path>example/path</path>
</read_file>

Now let me actually read the file you need:

<read_file>
<path>real/target/file.py</path>
</read_file>
'''
    result = parse_xml_tool(content)
    assert result is not None, "Should have found a tool call"
    assert result.name == "read_file"
    # Should get the LAST path, not the example
    assert result.parameters.get("path") == "real/target/file.py", \
        f"Got wrong path: {result.parameters.get('path')}"
    print("✓ test_parse_xml_tool_last_match passed")


def test_write_to_file_with_xml_content():
    """Test write_to_file tool parsing when content has XML examples."""
    content = '''
I'll create the file with the documentation:

<write_to_file>
<path>prompts.go</path>
<content>
package main

// Example usage:
// <read_file>
// <path>some/file</path>
// </read_file>

func getPrompt() string {
    return "hello"
}
</content>
</write_to_file>
'''
    result = parse_xml_tool(content)
    assert result is not None, "parse_xml_tool returned None"
    assert result.name == "write_to_file", f"Wrong tool: {result.name}"
    assert result.parameters.get("path") == "prompts.go", f"Wrong path: {result.parameters.get('path')}"
    # Content should include the XML examples
    content_param = result.parameters.get("content", "")
    assert "<read_file>" in content_param, f"Missing <read_file> in content"
    assert "getPrompt" in content_param, f"Missing getPrompt in content"
    print("✓ test_write_to_file_with_xml_content passed")


def test_replace_in_file_complex():
    """Test replace_in_file with complex diff containing XML."""
    content = '''
Let me update the file:

<replace_in_file>
<path>prompts.go</path>
<diff>
<<<<<<< SEARCH
## Tools

No tools defined yet.
=======
## Tools

### read_file
Read a file from disk:

<read_file>
<path>path/to/file</path>
</read_file>

### write_to_file
Write content to a file:

<write_to_file>
<path>path/to/file</path>
<content>
file content
</content>
</write_to_file>
>>>>>>> REPLACE
</diff>
</replace_in_file>
'''
    result = parse_xml_tool(content)
    assert result is not None, "parse_xml_tool returned None"
    assert result.name == "replace_in_file", f"Wrong tool: {result.name}"
    assert result.parameters.get("path") == "prompts.go", f"Wrong path: {result.parameters.get('path')}"
    
    diff = result.parameters.get("diff", "")
    blocks = parse_search_replace_blocks(diff)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    
    search, replace = blocks[0]
    assert "No tools defined yet" in search
    assert "<read_file>" in replace
    assert "<write_to_file>" in replace
    print("✓ test_replace_in_file_complex passed")


def test_file_creation_via_replace():
    """Test creating a file incrementally using SEARCH/REPLACE blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_prompts.go"
        
        # Step 1: Create initial file
        initial_content = '''package main

func getPrompt() string {
    return "TODO"
}
'''
        filepath.write_text(initial_content)
        
        # Step 2: Replace with fuller content
        diff = '''
<<<<<<< SEARCH
func getPrompt() string {
    return "TODO"
}
=======
func getPrompt() string {
    return `You are an AI assistant.

## Tools

### read_file
<read_file>
<path>file</path>
</read_file>
`
}
>>>>>>> REPLACE
'''
        blocks = parse_search_replace_blocks(diff)
        assert len(blocks) == 1
        
        search, replace = blocks[0]
        content = filepath.read_text()
        
        if search in content:
            content = content.replace(search, replace, 1)
            filepath.write_text(content)
        
        final = filepath.read_text()
        assert "<read_file>" in final
        assert "You are an AI assistant" in final
        print("✓ test_file_creation_via_replace passed")


def test_prompts_go_simulation():
    """Simulate building prompts.go via multiple SEARCH/REPLACE operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "prompts.go"
        
        # Initial skeleton
        filepath.write_text('''package main

const defaultPrompt = `TODO`

func getPrompt() string {
    return defaultPrompt
}
''')
        
        # Replace 1: Add header content
        diff1 = '''
<<<<<<< SEARCH
const defaultPrompt = `TODO`
=======
const defaultPrompt = `You are Cline, an expert AI coding assistant.

====

TOOL USE

You have access to tools formatted as XML:

<tool_name>
<param>value</param>
</tool_name>

====

TOOLS`
>>>>>>> REPLACE
'''
        
        content = filepath.read_text()
        blocks = parse_search_replace_blocks(diff1)
        for search, replace in blocks:
            if search in content:
                content = content.replace(search, replace, 1)
        filepath.write_text(content)
        
        # Verify
        result = filepath.read_text()
        assert "You are Cline" in result
        assert "<tool_name>" in result
        assert "<param>" in result
        
        print("✓ test_prompts_go_simulation passed")


def run_all_tests():
    """Run all tests."""
    print("\n=== Testing SEARCH/REPLACE Block Parsing ===\n")
    
    tests = [
        test_basic_search_replace,
        test_multiple_blocks,
        test_xml_content_in_blocks,
        test_parse_xml_tool_last_match,
        test_write_to_file_with_xml_content,
        test_replace_in_file_complex,
        test_file_creation_via_replace,
        test_prompts_go_simulation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
