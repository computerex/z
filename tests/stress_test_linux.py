#!/usr/bin/env python3
"""Stress test the harness executable in Linux environment."""

import subprocess
import sys
import os

HARNESS_PATH = "/app/harness"
WORKSPACE = "/app/torture_test"


def run_test(name: str, prompt: str, timeout: int = 120) -> tuple[bool, str]:
    """Run a single test and return (passed, output)."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [HARNESS_PATH, WORKSPACE, "--new"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/app",
            encoding='utf-8',
            errors='replace',
        )
        output = (result.stdout or "") + (result.stderr or "")
        print(f"Exit code: {result.returncode}")
        print(f"Output length: {len(output)} chars")
        
        # Show first 500 chars of output for debugging
        if len(output) > 0:
            preview = output[:500] + "..." if len(output) > 500 else output
            print(f"Preview: {preview}")
        
        return True, output
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, str(e)


def main():
    print(f"Testing harness executable: {HARNESS_PATH}")
    print(f"Workspace: {WORKSPACE}")
    
    # Check executable exists
    if not os.path.exists(HARNESS_PATH):
        print(f"ERROR: Executable not found at {HARNESS_PATH}")
        return 1
    
    # Check it's executable
    result = subprocess.run([HARNESS_PATH, "--help"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Executable failed to run --help")
        print(result.stderr)
        return 1
    print("Executable check: OK")
    
    tests = [
        # Test 1: Read non-existent file (error handling)
        ("Error handling - missing file", 
         "read the file nonexistent_file_12345.txt and tell me what's in it"),
        
        # Test 2: Command with no output
        ("Silent command",
         "run the command: echo"),
        
        # Test 3: List files
        ("List files",
         "list all files in the current directory"),
        
        # Test 4: Search for files
        ("Search for files",
         "search for files matching *.py in the current directory"),
        
        # Test 5: Multi-step task
        ("Multi-step task",
         "1) list the files in this directory, 2) read config.py, 3) tell me what DATABASE_URL is set to"),
        
        # Test 6: Write and verify
        ("Write and verify",
         "create a file called test_output.txt with the content 'Hello World' then read it back to verify"),
         
        # Test 7: Unicode content
        ("Unicode content",
         "create a file called unicode_test.txt containing: Hello 你好 🚀"),
        
        # Test 8: Command that exits with error code
        ("Error exit code",
         "run this command that will fail: python3 -c \"import nonexistent_module_xyz\""),
        
        # Test 9: Read a file
        ("Read file",
         "read the config.py file and summarize what it contains"),
        
        # Test 10: Rapid file ops
        ("Rapid file ops",
         "create 3 files: a.txt, b.txt, c.txt with contents '1', '2', '3' respectively, then read all 3 and sum the numbers"),
    ]
    
    passed = 0
    failed = 0
    
    for name, prompt in tests:
        success, output = run_test(name, prompt, timeout=90)
        if success:
            passed += 1
            print(f"✓ PASSED")
        else:
            failed += 1
            print(f"✗ FAILED")
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
