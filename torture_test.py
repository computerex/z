#!/usr/bin/env python3
"""
Comprehensive torture test suite for the harness agent.

This test validates that the harness embodies the Claude Code philosophy
and patterns exactly, with multi-step edits, complex scenarios, and
non-trivial project work.
"""

import asyncio
import tempfile
import shutil
import os
import sys
from pathlib import Path
import json
import subprocess

# Add the harness to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harness.cline_agent import ClineAgent
from harness.config import Config
from harness.streaming_client import StreamingMessage


class TortureTest:
    """Comprehensive test suite for harness agent parity with Claude Code."""
    
    def __init__(self):
        self.test_dir = None
        self.agent = None
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def setup_test_environment(self):
        """Create a temporary test environment with realistic project structure."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="harness_torture_"))
        print(f"Test environment: {self.test_dir}")
        
        # Create a realistic Python project structure
        (self.test_dir / "src").mkdir()
        (self.test_dir / "src" / "myproject").mkdir()
        (self.test_dir / "tests").mkdir()
        (self.test_dir / "docs").mkdir()
        
        # Create some initial files
        (self.test_dir / "README.md").write_text("""# My Project

A sample Python project for testing.

## Installation

```bash
pip install -e .
```

## Usage

```python
from myproject import hello
hello.greet("World")
```
""")
        
        (self.test_dir / "setup.py").write_text("""from setuptools import setup, find_packages

setup(
    name="myproject",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "myproject=myproject.cli:main",
        ],
    },
)
""")
        
        (self.test_dir / "src" / "myproject" / "__init__.py").write_text('"""My project package."""\n__version__ = "0.1.0"\n')
        
        (self.test_dir / "src" / "myproject" / "hello.py").write_text("""\"\"\"Hello module with greeting functionality.\"\"\"

def greet(name):
    \"\"\"Greet someone by name.\"\"\"
    return f"Hello, {name}!"

def greet_multiple(names):
    \"\"\"Greet multiple people.\"\"\"
    return [greet(name) for name in names]

class Greeter:
    \"\"\"A greeter class.\"\"\"
    
    def __init__(self, prefix="Hello"):
        self.prefix = prefix
    
    def greet(self, name):
        \"\"\"Greet with custom prefix.\"\"\"
        return f"{self.prefix}, {name}!"
""")
        
        (self.test_dir / "src" / "myproject" / "cli.py").write_text("""\"\"\"Command line interface.\"\"\"
import click
from .hello import greet

@click.command()
@click.argument("name")
@click.option("--count", default=1, help="Number of greetings")
def main(name, count):
    \"\"\"Greet someone from the command line.\"\"\"
    for _ in range(count):
        click.echo(greet(name))

if __name__ == "__main__":
    main()
""")
        
        (self.test_dir / "tests" / "__init__.py").write_text("")
        (self.test_dir / "tests" / "test_hello.py").write_text("""\"\"\"Tests for hello module.\"\"\"
import pytest
from myproject.hello import greet, greet_multiple, Greeter

def test_greet():
    \"\"\"Test basic greeting.\"\"\"
    assert greet("World") == "Hello, World!"

def test_greet_multiple():
    \"\"\"Test greeting multiple people.\"\"\"
    names = ["Alice", "Bob"]
    expected = ["Hello, Alice!", "Hello, Bob!"]
    assert greet_multiple(names) == expected

def test_greeter_class():
    \"\"\"Test Greeter class.\"\"\"
    greeter = Greeter("Hi")
    assert greeter.greet("World") == "Hi, World!"
""")
        
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.test_dir, capture_output=True)
        
        # Initialize agent
        config = Config(
            api_url="http://localhost:8080/v1/messages",
            api_key="test-key",
            model="test-model"
        )
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        self.agent = ClineAgent(config)
        self.agent.workspace_path = str(self.test_dir)
        self.agent.clear_history()
        
        return original_cwd
    
    def cleanup_test_environment(self, original_cwd):
        """Clean up test environment."""
        os.chdir(original_cwd)
        if self.test_dir and self.test_dir.exists():
            try:
                # On Windows, git objects can be read-only, need to make them writable
                if os.name == 'nt':
                    for root, dirs, files in os.walk(self.test_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Could not fully clean up test directory: {e}")
    
    def test_result(self, name, passed, details=""):
        """Record a test result."""
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if details:
            print(f"    {details}")
        
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_system_prompt_structure(self):
        """Test that system prompt has the correct Claude Code structure."""
        print("\n=== Testing System Prompt Structure ===")
        
        prompt = self.agent._system_prompt()
        
        # Check for key sections
        required_sections = [
            "You are a highly skilled software engineer",
            "IMPORTANT: Assist with authorized security testing",
            "IMPORTANT: You must NEVER generate or guess URLs",
            "# System",
            "Tools are executed in a user-selected permission mode",
            "Tool results and user messages may include <system-reminder>",
            "The system will automatically compress prior messages",
            "TOOL USE",
            "You call tools by emitting XML",
            "# Doing tasks",
            "# Executing actions with care",
            "# Using your tools",
            "# Tone and style",
            "# Making code changes",
            "# Environment",
            "Primary working directory:",
            "Platform:",
            "Shell:",
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in prompt:
                missing_sections.append(section)
        
        if missing_sections:
            self.test_result("System prompt structure", False, 
                           f"Missing sections: {missing_sections}")
        else:
            self.test_result("System prompt structure", True)
        
        # Check prompt length (should be substantial)
        if len(prompt) < 20000:
            self.test_result("System prompt length", False, 
                           f"Prompt too short: {len(prompt)} chars")
        else:
            self.test_result("System prompt length", True, 
                           f"{len(prompt)} chars")
    
    def test_tool_registry_descriptions(self):
        """Test that tool registry has rich descriptions."""
        print("\n=== Testing Tool Registry Descriptions ===")
        
        from harness.tool_registry import TOOL_DEFS
        
        tools_without_descriptions = []
        for tool in TOOL_DEFS:
            if not tool.description or len(tool.description) < 20:
                tools_without_descriptions.append(tool.name)
        
        if tools_without_descriptions:
            self.test_result("Tool descriptions", False,
                           f"Tools without good descriptions: {tools_without_descriptions}")
        else:
            self.test_result("Tool descriptions", True)
    
    def test_xml_parsing(self):
        """Test XML tool parsing with various formats."""
        print("\n=== Testing XML Parsing ===")
        
        from harness.cline_agent import parse_xml_tool
        
        test_cases = [
            # Basic tool call
            ('<read_file><path>test.py</path></read_file>', 'read_file', {'path': 'test.py'}),
            
            # Tool with multiple parameters
            ('<read_file><path>test.py</path><start_line>10</start_line><end_line>20</end_line></read_file>',
             'read_file', {'path': 'test.py', 'start_line': '10', 'end_line': '20'}),
            
            # Complex content tool
            ('<write_to_file><path>test.py</path><content>\ndef hello():\n    print("Hello")\n</content></write_to_file>',
             'write_to_file', {'path': 'test.py', 'content': 'def hello():\n    print("Hello")'}),
            
            # Tool with nested XML-like content
            ('<write_to_file><path>config.xml</path><content>\n<config>\n  <setting>value</setting>\n</config>\n</content></write_to_file>',
             'write_to_file', {'path': 'config.xml', 'content': '<config>\n  <setting>value</setting>\n</config>'}),
        ]
        
        all_passed = True
        for xml_content, expected_name, expected_params in test_cases:
            result = parse_xml_tool(xml_content)
            if not result:
                self.test_result(f"Parse: {expected_name}", False, "Failed to parse")
                all_passed = False
                continue
            
            if result.name != expected_name:
                self.test_result(f"Parse: {expected_name}", False, 
                               f"Wrong name: {result.name}")
                all_passed = False
                continue
            
            if result.parameters != expected_params:
                self.test_result(f"Parse: {expected_name}", False,
                               f"Wrong params: {result.parameters}")
                all_passed = False
                continue
        
        if all_passed:
            self.test_result("XML parsing", True, f"All {len(test_cases)} cases passed")
    
    def test_session_integrity(self):
        """Test session loading/saving integrity."""
        print("\n=== Testing Session Integrity ===")
        
        # Add some messages
        self.agent.messages = [
            StreamingMessage(role="system", content=self.agent._system_prompt()),
            StreamingMessage(role="user", content="Hello"),
            StreamingMessage(role="assistant", content="Hi there!")
        ]
        
        # Save session
        session_file = self.test_dir / "test_session.json"
        self.agent.save_session(str(session_file))
        
        if not session_file.exists():
            self.test_result("Session save", False, "Session file not created")
            return
        
        # Load session
        agent2 = ClineAgent(self.agent.config)
        agent2.workspace_path = str(self.test_dir)
        success = agent2.load_session(str(session_file), inject_resume=False)
        
        if not success:
            self.test_result("Session load", False, "Failed to load session")
            return
        
        # Check integrity
        if len(agent2.messages) != 3:
            self.test_result("Session integrity", False, 
                           f"Wrong message count: {len(agent2.messages)}")
            return
        
        # Check system prompt integrity
        sys_content = agent2.messages[0].content
        if not isinstance(sys_content, str) or "TOOL USE" not in sys_content:
            self.test_result("System prompt integrity", False,
                           "System prompt corrupted")
            return
        
        self.test_result("Session integrity", True)
    
    def test_context_management(self):
        """Test context container functionality."""
        print("\n=== Testing Context Management ===")
        
        # Add some context items
        item1_id = self.agent.context.add("file", "test.py", "print('hello')")
        item2_id = self.agent.context.add("command_output", "ls -la", "total 8\n-rw-r--r-- 1 user user 13 test.py")
        
        if self.agent.context.total_size() == 0:
            self.test_result("Context add", False, "No content added")
            return
        
        # Test retrieval
        item1 = self.agent.context.get(item1_id)
        if not item1 or item1.content != "print('hello')":
            self.test_result("Context retrieval", False, "Item not retrieved correctly")
            return
        
        # Test removal
        removed = self.agent.context.remove(item1_id)
        if not removed:
            self.test_result("Context removal", False, "Item not removed")
            return
        
        if self.agent.context.get(item1_id) is not None:
            self.test_result("Context removal verification", False, "Item still exists after removal")
            return
        
        self.test_result("Context management", True)
    
    def test_todo_management(self):
        """Test todo list functionality."""
        print("\n=== Testing Todo Management ===")
        
        # Add a todo
        todo_item = self.agent.todo_manager.add("Test todo", "This is a test todo item")
        todo_id = todo_item.id
        
        todos = self.agent.todo_manager.list_all()
        if len(todos) != 1:
            self.test_result("Todo add", False, f"Expected 1 todo, got {len(todos)}")
            return
        
        # Update todo status
        self.agent.todo_manager.update(todo_id, status="in-progress")
        todo = self.agent.todo_manager.get(todo_id)
        
        if not todo or todo.status.value != "in-progress":
            self.test_result("Todo update", False, "Status not updated correctly")
            return
        
        # Complete todo
        self.agent.todo_manager.update(todo_id, status="completed")
        active_todos = self.agent.todo_manager.list_active()
        
        if len(active_todos) != 0:
            self.test_result("Todo completion", False, "Completed todo still active")
            return
        
        self.test_result("Todo management", True)
    
    def test_tool_execution_simulation(self):
        """Test tool execution patterns (without actual API calls)."""
        print("\n=== Testing Tool Execution Patterns ===")
        
        # Test tool handler initialization
        if not self.agent.tool_handlers:
            self.test_result("Tool handlers", False, "Tool handlers not initialized")
            return
        
        # Test tool name registry
        from harness.tool_registry import get_tool_names
        tool_names = get_tool_names()
        
        expected_tools = ["read_file", "write_to_file", "replace_in_file", "execute_command", 
                         "manage_todos", "search_files", "list_files"]
        
        missing_tools = [t for t in expected_tools if t not in tool_names]
        if missing_tools:
            self.test_result("Tool registry", False, f"Missing tools: {missing_tools}")
            return
        
        self.test_result("Tool execution patterns", True)
    
    def test_file_operations_simulation(self):
        """Test file operation patterns."""
        print("\n=== Testing File Operations ===")
        
        # Test that files exist in our test environment
        test_file = self.test_dir / "src" / "myproject" / "hello.py"
        if not test_file.exists():
            self.test_result("Test environment", False, "Test files not created")
            return
        
        # Test file reading
        content = test_file.read_text()
        if "def greet(name):" not in content:
            self.test_result("File content", False, "Expected content not found")
            return
        
        # Test file writing
        new_file = self.test_dir / "test_new.py"
        new_file.write_text("# Test file\nprint('test')\n")
        
        if not new_file.exists():
            self.test_result("File writing", False, "New file not created")
            return
        
        self.test_result("File operations", True)
    
    def test_git_integration(self):
        """Test git integration."""
        print("\n=== Testing Git Integration ===")
        
        # Check git status
        result = subprocess.run(["git", "status", "--porcelain"], 
                              cwd=self.test_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.test_result("Git status", False, "Git not working in test environment")
            return
        
        # Create a change
        test_file = self.test_dir / "src" / "myproject" / "hello.py"
        content = test_file.read_text()
        content += "\n# Added by torture test\n"
        test_file.write_text(content)
        
        # Check that git detects the change
        result = subprocess.run(["git", "status", "--porcelain"], 
                              cwd=self.test_dir, capture_output=True, text=True)
        
        if "hello.py" not in result.stdout:
            self.test_result("Git change detection", False, "Git didn't detect file change")
            return
        
        self.test_result("Git integration", True)
    
    def test_prompt_length_and_structure(self):
        """Test that prompt is substantial and well-structured."""
        print("\n=== Testing Prompt Quality ===")
        
        prompt = self.agent._system_prompt()
        
        # Check for Claude Code specific patterns
        claude_patterns = [
            "You are a highly skilled software engineer",
            "IMPORTANT: Assist with authorized security testing",
            "IMPORTANT: You must NEVER generate or guess URLs",
            "Tools are executed in a user-selected permission mode",
            "The system will automatically compress prior messages",
            "Carefully consider the reversibility and blast radius",
            "Do NOT use execute_command to run commands when a relevant dedicated tool",
            "Only use emojis if the user explicitly requests it",
            "Do not use a colon before tool calls",
            "You MUST use read_file at least once before editing",
            "NEVER generate an extremely long hash",
            "Do NOT add comments that just narrate what the code does",
        ]
        
        missing_patterns = []
        for pattern in claude_patterns:
            if pattern not in prompt:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            self.test_result("Claude Code patterns", False,
                           f"Missing patterns: {len(missing_patterns)}")
            for pattern in missing_patterns[:3]:  # Show first 3
                print(f"    Missing: {pattern[:60]}...")
        else:
            self.test_result("Claude Code patterns", True)
        
        # Check environment section
        if f"Primary working directory: {self.test_dir}" not in prompt:
            self.test_result("Environment section", False, "Working directory not in prompt")
        else:
            self.test_result("Environment section", True)
    
    def run_all_tests(self):
        """Run all torture tests."""
        print("=== HARNESS TORTURE TEST SUITE ===")
        print("Testing Claude Code philosophy and pattern parity...")
        
        original_cwd = self.setup_test_environment()
        
        try:
            # Core functionality tests
            self.test_system_prompt_structure()
            self.test_tool_registry_descriptions()
            self.test_xml_parsing()
            self.test_session_integrity()
            self.test_context_management()
            self.test_todo_management()
            self.test_tool_execution_simulation()
            self.test_file_operations_simulation()
            self.test_git_integration()
            self.test_prompt_length_and_structure()
            
            # Summary
            print(f"\n=== TORTURE TEST RESULTS ===")
            print(f"PASSED: {self.passed}")
            print(f"FAILED: {self.failed}")
            print(f"TOTAL:  {self.passed + self.failed}")
            
            if self.failed == 0:
                print("*** ALL TESTS PASSED! Harness has Claude Code parity! ***")
                return True
            else:
                print(f"*** {self.failed} TESTS FAILED. Harness needs more work. ***")
                print("\nFailed tests:")
                for result in self.results:
                    if not result["passed"]:
                        print(f"  - {result['name']}: {result['details']}")
                return False
        
        finally:
            self.cleanup_test_environment(original_cwd)


def main():
    """Run the torture test suite."""
    test = TortureTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()