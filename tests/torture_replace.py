#!/usr/bin/env python3
"""
Torture test suite specifically for replace_in_file operations.

This tests the most error-prone and difficult operation: editing files.
"""

import subprocess
import sys
import os
import tempfile
import shutil

# Detect environment
if os.path.exists("/app/harness"):
    HARNESS_PATH = "/app/harness"
    BASE_DIR = "/app"
else:
    HARNESS_PATH = sys.executable
    HARNESS_ARGS = [os.path.join(os.path.dirname(__file__), "..", "harness.py")]
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_harness(workspace: str, prompt: str, timeout: int = 120) -> tuple[int, str]:
    """Run harness with a prompt and return (exit_code, output)."""
    if os.path.exists("/app/harness"):
        cmd = [HARNESS_PATH, workspace, "--new"]
    else:
        cmd = [HARNESS_PATH, *HARNESS_ARGS, workspace, "--new"]
    
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=BASE_DIR,
            encoding='utf-8',
            errors='replace',
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, str(e)


def create_test_file(workspace: str, filename: str, content: str) -> str:
    """Create a test file with given content."""
    path = os.path.join(workspace, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def read_test_file(workspace: str, filename: str) -> str:
    """Read a test file."""
    path = os.path.join(workspace, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def file_exists(workspace: str, filename: str) -> bool:
    """Check if file exists."""
    return os.path.exists(os.path.join(workspace, filename))


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        s = f"{status}: {self.name}"
        if self.error:
            s += f"\n  Error: {self.error}"
        if self.details and not self.passed:
            s += f"\n  Details: {self.details[:200]}"
        return s


def run_replace_test(workspace: str, name: str, setup_content: str, prompt: str, 
                     expected_contains: list[str] = None, 
                     expected_not_contains: list[str] = None,
                     filename: str = "test_file.py") -> TestResult:
    """Run a single replace test."""
    result = TestResult(name)
    print(f"  Running: {name}...", end=" ", flush=True)
    
    try:
        # Cleanup workspace before test - remove all files except hidden
        for f in os.listdir(workspace):
            path = os.path.join(workspace, f)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        
        # Setup: create the test file
        create_test_file(workspace, filename, setup_content)
        
        # Run the harness
        exit_code, output = run_harness(workspace, prompt)
        
        # Check if file still exists
        if not file_exists(workspace, filename):
            result.error = "File was deleted!"
            print("FAIL", flush=True)
            return result
        
        # Read the result
        final_content = read_test_file(workspace, filename)
        result.details = f"Final content:\n{final_content[:300]}"
        
        # Check expected content
        if expected_contains:
            for expected in expected_contains:
                if expected not in final_content:
                    result.error = f"Expected '{expected}' not found in result"
                    print("FAIL", flush=True)
                    return result
        
        if expected_not_contains:
            for not_expected in expected_not_contains:
                if not_expected in final_content:
                    result.error = f"Unexpected '{not_expected}' still present"
                    print("FAIL", flush=True)
                    return result
        
        result.passed = True
        print("PASS", flush=True)
        
    except Exception as e:
        result.error = str(e)
        print("ERROR", flush=True)
    
    return result


def main():
    print("=" * 70)
    print("REPLACE_IN_FILE TORTURE TEST SUITE")
    print("=" * 70)
    
    # Create temporary workspace
    workspace = tempfile.mkdtemp(prefix="harness_replace_test_")
    print(f"Workspace: {workspace}\n")
    
    results = []
    
    try:
        # ============================================================
        # TEST 1: Simple single-line replacement
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Simple single-line replacement",
            setup_content='name = "old_value"\nage = 25\n',
            prompt='In test_file.py, change "old_value" to "new_value"',
            expected_contains=['"new_value"'],
            expected_not_contains=['"old_value"'],
        ))
        
        # ============================================================
        # TEST 2: Multi-line block replacement
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Multi-line block replacement",
            setup_content='''def hello():
    print("Hello")
    print("World")

def goodbye():
    print("Bye")
''',
            prompt='In test_file.py, change the hello function to print "Greetings" instead of "Hello"',
            expected_contains=['print("Greetings")'],
            expected_not_contains=['print("Hello")'],
        ))
        
        # ============================================================
        # TEST 3: Preserve indentation
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Preserve indentation",
            setup_content='''class MyClass:
    def method(self):
        if True:
            x = 1
            y = 2
        return x + y
''',
            prompt='In test_file.py, change "x = 1" to "x = 100" while preserving the indentation',
            expected_contains=['            x = 100'],
            expected_not_contains=['x = 1\n'],
        ))
        
        # ============================================================
        # TEST 4: Replace with more lines than original
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace with more lines (expansion)",
            setup_content='''# Header
single_line = True
# Footer
''',
            prompt='In test_file.py, replace "single_line = True" with three lines: "line1 = 1", "line2 = 2", "line3 = 3"',
            expected_contains=['line1 = 1', 'line2 = 2', 'line3 = 3'],
            expected_not_contains=['single_line = True'],
        ))
        
        # ============================================================
        # TEST 5: Replace with fewer lines (contraction)
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace with fewer lines (contraction)",
            setup_content='''# Start
a = 1
b = 2
c = 3
# End
''',
            prompt='In test_file.py, replace the three lines "a = 1", "b = 2", "c = 3" with just "combined = 6"',
            expected_contains=['combined = 6'],
            expected_not_contains=['a = 1', 'b = 2', 'c = 3'],
        ))
        
        # ============================================================
        # TEST 6: Replace in middle of file (preserve before/after)
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in middle preserving context",
            setup_content='''line1 = "unchanged"
line2 = "unchanged"
target = "CHANGE_ME"
line4 = "unchanged"
line5 = "unchanged"
''',
            prompt='In test_file.py, change target = "CHANGE_ME" to target = "CHANGED"',
            expected_contains=['line1 = "unchanged"', 'target = "CHANGED"', 'line5 = "unchanged"'],
            expected_not_contains=['CHANGE_ME'],
        ))
        
        # ============================================================
        # TEST 7: Handle special characters in content
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Special characters (quotes, backslashes)",
            setup_content='''path = "C:\\Users\\old"
regex = r"\\d+"
''',
            prompt='In test_file.py, change the path from "C:\\Users\\old" to "C:\\Users\\new"',
            expected_contains=['Users\\new'],
            expected_not_contains=['Users\\old'],
        ))
        
        # ============================================================
        # TEST 8: Unicode content
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Unicode characters",
            setup_content='''greeting = "Hello"
emoji = "😀"
chinese = "你好"
''',
            prompt='In test_file.py, change greeting = "Hello" to greeting = "Bonjour"',
            expected_contains=['greeting = "Bonjour"', 'emoji = "😀"', 'chinese = "你好"'],
            expected_not_contains=['greeting = "Hello"'],
        ))
        
        # ============================================================
        # TEST 9: Empty lines handling
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Preserve empty lines",
            setup_content='''def func1():
    pass


def func2():
    pass
''',
            prompt='In test_file.py, change "def func1():" to "def new_func1():"',
            expected_contains=['def new_func1():', '\n\n\ndef func2():'],
        ))
        
        # ============================================================
        # TEST 10: Replace function body
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace entire function body",
            setup_content='''def calculate(x):
    result = x * 2
    return result

def other():
    pass
''',
            prompt='In test_file.py, change the calculate function body to: result = x * 10; return result',
            expected_contains=['x * 10'],
            expected_not_contains=['x * 2'],
        ))
        
        # ============================================================
        # TEST 11: Multiple occurrences - replace first only
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Multiple occurrences - specific replacement",
            setup_content='''x = 1
x = 1
x = 1
''',
            prompt='In test_file.py, there are three lines with "x = 1". Change only the FIRST one to "x = 100"',
            expected_contains=['x = 100', 'x = 1'],
        ))
        
        # ============================================================
        # TEST 12: Replace import statement
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace import statement",
            setup_content='''import os
import sys
from pathlib import Path

def main():
    pass
''',
            prompt='In test_file.py, change "import os" to "import os.path"',
            expected_contains=['import os.path', 'import sys'],
        ))
        
        # ============================================================
        # TEST 13: Replace class definition
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace class definition line",
            setup_content='''class OldName:
    def __init__(self):
        self.value = 0
''',
            prompt='In test_file.py, rename the class from OldName to NewName',
            expected_contains=['class NewName:'],
            expected_not_contains=['class OldName:'],
        ))
        
        # ============================================================
        # TEST 14: Add new code after existing line
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Expand: add lines after existing",
            setup_content='''def setup():
    config = {}
    return config
''',
            prompt='In test_file.py, after "config = {}" add a new line "config[\"debug\"] = True"',
            expected_contains=['config = {}', 'config["debug"] = True'],
        ))
        
        # ============================================================
        # TEST 15: Handle trailing whitespace
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Handle trailing whitespace",
            setup_content='value = 1   \nother = 2\n',
            prompt='In test_file.py, change "value = 1" (which may have trailing spaces) to "value = 999"',
            expected_contains=['value = 999'],
            expected_not_contains=['value = 1'],
        ))
        
        # ============================================================
        # TEST 16: Replace in JSON-like content
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "JSON-like content replacement",
            setup_content='''{
    "name": "old_name",
    "version": "1.0.0",
    "description": "A test"
}
''',
            prompt='In test_file.json, change "name": "old_name" to "name": "new_name"',
            expected_contains=['"name": "new_name"'],
            expected_not_contains=['"name": "old_name"'],
            filename="test_file.json",
        ))
        
        # ============================================================
        # TEST 17: Replace comment
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace comment content",
            setup_content='''# TODO: fix this bug
def buggy():
    pass
''',
            prompt='In test_file.py, change the comment "# TODO: fix this bug" to "# DONE: bug fixed"',
            expected_contains=['# DONE: bug fixed'],
            expected_not_contains=['# TODO: fix this bug'],
        ))
        
        # ============================================================
        # TEST 18: Large file simulation (many lines)
        # ============================================================
        large_content = '\n'.join([f'line_{i} = {i}' for i in range(100)])
        large_content += '\ntarget_line = "FIND_ME"\n'
        large_content += '\n'.join([f'line_{i} = {i}' for i in range(100, 200)])
        
        results.append(run_replace_test(
            workspace,
            "Large file - find and replace in middle",
            setup_content=large_content,
            prompt='In test_file.py, find "target_line = \\"FIND_ME\\"" and change it to "target_line = \\"FOUND\\""',
            expected_contains=['target_line = "FOUND"', 'line_0 = 0', 'line_199 = 199'],
            expected_not_contains=['target_line = "FIND_ME"'],
        ))
        
        # ============================================================
        # TEST 19: Replace with exact whitespace matching
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Exact whitespace matching",
            setup_content='''if True:
\tindented_with_tab = 1
    indented_with_spaces = 2
''',
            prompt='In test_file.py, change "indented_with_spaces = 2" to "indented_with_spaces = 200"',
            expected_contains=['indented_with_spaces = 200'],
            expected_not_contains=['indented_with_spaces = 2\n'],
        ))
        
        # ============================================================
        # TEST 20: Don't corrupt file on failed match
        # ============================================================
        original_content = '''preserved = True
intact = True
'''
        results.append(run_replace_test(
            workspace,
            "Graceful handling of no-match",
            setup_content=original_content,
            prompt='In test_file.py, try to replace "nonexistent_string_xyz" with "something". The file should remain unchanged.',
            expected_contains=['preserved = True', 'intact = True'],
        ))
        
        # ============================================================
        # TEST 21: Nested structures (deeply indented)
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Deeply nested indentation",
            setup_content='''def outer():
    def inner():
        if True:
            for i in range(10):
                while True:
                    value = "old"
                    break
''',
            prompt='In test_file.py, change value = "old" to value = "new" while preserving the deep indentation',
            expected_contains=['                    value = "new"'],
            expected_not_contains=['value = "old"'],
        ))
        
        # ============================================================
        # TEST 22: Replace with regex-like content
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Content that looks like regex",
            setup_content='''pattern = r"^.*$"
replacement = r"\\1\\2"
''',
            prompt='In test_file.py, change pattern = r"^.*$" to pattern = r"^\\w+$"',
            expected_contains=[r'pattern = r"^\w+$"'],
        ))
        
        # ============================================================
        # TEST 23: Multi-line string literal
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Multi-line string literal",
            setup_content='''doc = """
This is a
multi-line
string
"""
''',
            prompt='In test_file.py, change "multi-line" to "multiline" in the docstring',
            expected_contains=['multiline'],
            expected_not_contains=['multi-line'],
        ))
        
        # ============================================================
        # TEST 24: Adjacent similar lines
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Adjacent similar lines - precise targeting",
            setup_content='''data1 = process(input1)
data2 = process(input2)
data3 = process(input3)
''',
            prompt='In test_file.py, change ONLY "data2 = process(input2)" to "data2 = transform(input2)"',
            expected_contains=['data1 = process(input1)', 'data2 = transform(input2)', 'data3 = process(input3)'],
        ))
        
        # ============================================================
        # TEST 25: Replace decorator
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace decorator",
            setup_content='''@old_decorator
def my_function():
    pass
''',
            prompt='In test_file.py, change @old_decorator to @new_decorator(param=True)',
            expected_contains=['@new_decorator(param=True)'],
            expected_not_contains=['@old_decorator'],
        ))
        
        # ============================================================
        # TEST 26: Type hints
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace type hints",
            setup_content='''def func(x: int, y: str) -> bool:
    return True
''',
            prompt='In test_file.py, change the return type from bool to Optional[bool]',
            expected_contains=['-> Optional[bool]:'],
            expected_not_contains=['-> bool:'],
        ))
        
        # ============================================================
        # TEST 27: List comprehension
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in list comprehension",
            setup_content='''result = [x * 2 for x in range(10) if x > 5]
''',
            prompt='In test_file.py, change "x * 2" in the comprehension to "x * 3"',
            expected_contains=['x * 3 for x in range(10)'],
            expected_not_contains=['x * 2 for'],
        ))
        
        # ============================================================
        # TEST 28: Error handling block
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in try/except block",
            setup_content='''try:
    risky_operation()
except ValueError:
    handle_error()
except Exception:
    pass
''',
            prompt='In test_file.py, change "except ValueError:" to "except (ValueError, TypeError):"',
            expected_contains=['except (ValueError, TypeError):'],
            expected_not_contains=['except ValueError:'],
        ))
        
        # ============================================================
        # TEST 29: Dictionary literal
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in dictionary literal",
            setup_content='''config = {
    "host": "localhost",
    "port": 8080,
    "debug": False,
}
''',
            prompt='In test_file.py, change "port": 8080 to "port": 9000',
            expected_contains=['"port": 9000'],
            expected_not_contains=['"port": 8080'],
        ))
        
        # ============================================================
        # TEST 30: Lambda expression
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace lambda expression",
            setup_content='''sorter = lambda x: x.lower()
''',
            prompt='In test_file.py, change the lambda to "lambda x: x.upper()"',
            expected_contains=['lambda x: x.upper()'],
            expected_not_contains=['lambda x: x.lower()'],
        ))
        
        # ============================================================
        # TEST 31: F-string
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in f-string",
            setup_content='''msg = f"Hello {name}, you are {age} years old"
''',
            prompt='In test_file.py, change "Hello" to "Hi" in the f-string',
            expected_contains=['f"Hi {name}'],
            expected_not_contains=['f"Hello {name}'],
        ))
        
        # ============================================================
        # TEST 32: Chained method calls
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace in chained method calls",
            setup_content='''result = (data
    .filter(x > 0)
    .map(double)
    .reduce(sum))
''',
            prompt='In test_file.py, change ".map(double)" to ".map(triple)"',
            expected_contains=['.map(triple)'],
            expected_not_contains=['.map(double)'],
        ))
        
        # ============================================================
        # TEST 33: Context manager
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace context manager",
            setup_content='''with open("file.txt", "r") as f:
    content = f.read()
''',
            prompt='In test_file.py, change open("file.txt", "r") to open("file.txt", "rb")',
            expected_contains=['open("file.txt", "rb")'],
            expected_not_contains=['open("file.txt", "r")'],
        ))
        
        # ============================================================
        # TEST 34: Async function
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace async function",
            setup_content='''async def fetch_data():
    result = await client.get("/api/old")
    return result
''',
            prompt='In test_file.py, change "/api/old" to "/api/new"',
            expected_contains=['"/api/new"'],
            expected_not_contains=['"/api/old"'],
        ))
        
        # ============================================================
        # TEST 35: Global/nonlocal statements
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace near global statement",
            setup_content='''counter = 0

def increment():
    global counter
    counter += 1
''',
            prompt='In test_file.py, change "counter += 1" to "counter += 100"',
            expected_contains=['counter += 100'],
            expected_not_contains=['counter += 1\n'],
        ))
        
        # ============================================================
        # TEST 36: SQL-like string
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Replace SQL query string",
            setup_content='''query = """
SELECT * FROM users
WHERE status = 'active'
ORDER BY created_at DESC
"""
''',
            prompt='In test_file.py, change "status = \'active\'" to "status = \'pending\'"',
            expected_contains=["status = 'pending'"],
            expected_not_contains=["status = 'active'"],
        ))
        
        # ============================================================
        # TEST 37: Consecutive edits simulation
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Consecutive file edit",
            setup_content='''a = 1
b = 2
c = 3
''',
            prompt='In test_file.py, change "a = 1" to "a = 100"',
            expected_contains=['a = 100', 'b = 2', 'c = 3'],
            expected_not_contains=['a = 1\n'],
        ))
        
        # ============================================================
        # TEST 38: Very long line
        # ============================================================
        long_line = "x = " + "a" * 500
        results.append(run_replace_test(
            workspace,
            "Very long line replacement",
            setup_content=f'''{long_line}
y = 2
''',
            prompt='In test_file.py, change the line starting with "x = aaaa..." to "x = 1"',
            expected_contains=['x = 1', 'y = 2'],
            expected_not_contains=['aaaa'],
        ))
        
        # ============================================================
        # TEST 39: Windows line endings (CRLF)
        # ============================================================
        # Clean workspace first
        for f in os.listdir(workspace):
            p = os.path.join(workspace, f)
            if os.path.isfile(p):
                os.remove(p)
        
        crlf_content = "line1 = 1\r\nline2 = 2\r\nline3 = 3\r\n"
        # Write with binary mode to preserve CRLF
        path = os.path.join(workspace, "test_file.py")
        with open(path, 'wb') as f:
            f.write(crlf_content.encode('utf-8'))
        
        result39 = TestResult("CRLF line endings")
        print(f"  Running: CRLF line endings...", end=" ", flush=True)
        try:
            exit_code, output = run_harness(workspace,
                'In test_file.py, change "line2 = 2" to "line2 = 200"')
            content = read_test_file(workspace, "test_file.py")
            if "line2 = 200" in content:
                result39.passed = True
                print("PASS", flush=True)
            else:
                result39.error = "CRLF content not properly edited"
                result39.details = repr(content)
                print("FAIL", flush=True)
        except Exception as e:
            result39.error = str(e)
            print("ERROR", flush=True)
        results.append(result39)
        
        # ============================================================
        # TEST 40: Binary-like content (should still work as text)
        # ============================================================
        results.append(run_replace_test(
            workspace,
            "Content with null-like escaped chars",
            setup_content='''data = b"\\x00\\x01\\x02"
other = "normal"
''',
            prompt='In test_file.py, change other = "normal" to other = "changed"',
            expected_contains=['other = "changed"'],
            expected_not_contains=['other = "normal"'],
        ))
        
    finally:
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        
        for r in results:
            print(r)
        
        print("\n" + "=" * 70)
        print(f"TOTAL: {passed} passed, {failed} failed out of {len(results)} tests")
        print("=" * 70)
        
        # Cleanup
        shutil.rmtree(workspace, ignore_errors=True)
        
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
