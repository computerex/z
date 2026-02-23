#!/usr/bin/env python3
"""Stress test the Python harness with challenging scenarios.

NOTE: These tests require a live LLM API connection. They validate:
1. The harness doesn't crash (basic smoke test)
2. The model actually generates tool calls (tool-use validation)
3. Python tracebacks don't appear (no code bugs)
"""

import subprocess
import sys
import time
import os
import re

HARNESS_PATH = os.path.join(os.path.dirname(__file__), "..", "harness.py")
WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "torture_test")

# Indicators that tool use actually happened (from ClineAgent._execute_tool console output)
TOOL_USE_INDICATORS = [
    "> Reading:",      # read_file
    "> Writing:",      # write_to_file
    "> Editing:",      # replace_in_file
    "> Running:",      # execute_command
    "> Listing:",      # list_files
    "> Searching:",    # search_files
    "> Todo:",         # manage_todos
    "> Web Search:",   # web_search
    "> Analyzing",     # analyze_image
    "[read_file result]",      # tool result in conversation
    "[write_to_file result]",  # tool result in conversation
    "[execute_command result]", # tool result in conversation
    "[list_files result]",     # tool result in conversation
    "[search_files result]",   # tool result in conversation
]

def check_tool_use(output: str) -> tuple[bool, list[str]]:
    """Check if the model actually used any tools.
    
    Returns (used_tools: bool, tool_indicators_found: list[str])
    """
    found = []
    for indicator in TOOL_USE_INDICATORS:
        if indicator in output:
            found.append(indicator)
    return len(found) > 0, found


def run_test(name: str, prompt: str, timeout: int = 120, 
             expect_tool_use: bool = True) -> tuple[bool, str]:
    """Run a single test and return (passed, output).
    
    Args:
        expect_tool_use: If True, test FAILS when model doesn't use tools.
                        This catches the "I can't run commands" failure mode.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, HARNESS_PATH, WORKSPACE, "--new"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(HARNESS_PATH),
            encoding='utf-8',
            errors='replace',
        )
        output = (result.stdout or "") + (result.stderr or "")
        print(f"Exit code: {result.returncode}")
        print(f"Output length: {len(output)} chars")
        
        # Check 1: Python tracebacks = code bug = FAIL
        if "Traceback (most recent call last)" in output:
            # Extract the actual traceback for diagnosis
            tb_match = re.search(r'Traceback \(most recent call last\).*?(?=\n\S|\Z)', 
                               output, re.DOTALL)
            tb_text = tb_match.group(0)[:500] if tb_match else "unknown"
            print(f"✗ CRASH: Python traceback detected!")
            print(f"  {tb_text[:200]}")
            return False, output
        
        # Check 2: Did the model use tools?
        used_tools, indicators = check_tool_use(output)
        if expect_tool_use and not used_tools:
            print(f"✗ NO TOOL USE: Model responded without using any tools!")
            print(f"  This means the LLM ignored its system prompt.")
            # Show first 200 chars of model output for diagnosis
            preview = output[:300].replace('\n', ' ')
            print(f"  Output preview: {preview}")
            return False, output
        
        if used_tools:
            print(f"  Tools used: {', '.join(indicators[:5])}")
        
        return True, output
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, str(e)


def main():
    tests = [
        # Test 1: Read non-existent file (error handling)
        ("Error handling - missing file", 
         "read the file nonexistent_file_12345.txt and tell me what's in it"),
        
        # Test 2: Command with no output
        ("Silent command",
         "run the command: echo."),
        
        # Test 3: Command that produces lots of output  
        ("Large output command",
         "run: dir /s C:\\Windows\\System32 | head -100"),
        
        # Test 4: Search with regex special chars
        ("Regex special chars in search",
         "search for files matching *.py in the current directory"),
        
        # Test 5: Multi-step task
        ("Multi-step task",
         "1) list the files in this directory, 2) read config.py, 3) tell me what DATABASE_URL is set to"),
        
        # Test 6: Write and verify
        ("Write and verify",
         "create a file called test_output.txt with the content 'Hello World' then read it back to verify"),
         
        # Test 7: Deeply nested path
        ("Deep path traversal",
         "what files are in torture_test folder?"),
        
        # Test 8: Unicode in file content
        ("Unicode content",
         "create a file called unicode_test.txt containing: 你好世界 🚀 αβγ"),
        
        # Test 9: Very long single line
        ("Long line handling",
         "create a file called longline.txt with a single line of 500 'x' characters"),
        
        # Test 10: Empty directory listing
        ("Empty dir",
         "create an empty directory called empty_test_dir and list its contents"),
        
        # Test 11: Command that exits with error code
        ("Error exit code",
         "run this command that will fail: python -c \"import nonexistent_module_xyz\""),
        
        # Test 12: Binary file detection
        ("Binary file",
         "try to read harness.py - just tell me how many lines it has"),
        
        # Test 13: Path with spaces 
        ("Path with spaces",
         "create a directory called 'test dir with spaces' and create a file inside it called 'my file.txt' with content 'test'"),
        
        # Test 14: Rapid file ops
        ("Rapid file ops",
         "create 3 files: a.txt, b.txt, c.txt with contents '1', '2', '3' respectively, then read all 3 and sum the numbers"),
    ]
    
    passed = 0
    failed = 0
    no_tool_use = 0
    
    for name, prompt in tests:
        success, output = run_test(name, prompt, timeout=90)
        if success:
            passed += 1
            print(f"✓ PASSED")
        else:
            failed += 1
            used_tools, _ = check_tool_use(output) if output != "TIMEOUT" else (False, [])
            if not used_tools:
                no_tool_use += 1
            print(f"✗ FAILED")
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    if no_tool_use > 0:
        print(f"WARNING: {no_tool_use} test(s) failed because the model did NOT use tools!")
        print(f"  This usually means the LLM is ignoring its system prompt.")
        print(f"  Check: model config, API key, system prompt length.")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
