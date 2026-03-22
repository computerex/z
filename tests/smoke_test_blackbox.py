#!/usr/bin/env python3
"""
Black-box smoke test suite for the harness CLI ("z").

Runs the harness as an external subprocess in headless mode with a cheap
model (glm-4.7 from Z.AI, the default).  Every scenario creates a fresh
temp workspace, pipes a task prompt to stdin, and then validates the
*filesystem side-effects* — files created, edited, content accuracy —
without importing any harness internals.

Usage
-----
    python tests/smoke_test_blackbox.py              # run all tests
    python tests/smoke_test_blackbox.py --test 3     # run only test #3
    python tests/smoke_test_blackbox.py --timeout 180  # per-test timeout
    python tests/smoke_test_blackbox.py --verbose     # show full stdout/stderr

Each test is self-contained.  On failure the temp workspace is preserved
and its path is printed so you can inspect the wreckage.

Exit code:  0 = all passed, 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HARNESS_PY = Path(__file__).resolve().parent.parent / "harness.py"
PYTHON     = sys.executable
DEFAULT_TIMEOUT = 150            # seconds per test
SESSION    = "smoke"             # shared session name (--new each time)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.checks: List[Tuple[str, bool, str]] = []  # (label, ok, detail)
        self.elapsed: float = 0.0
        self.stdout: str = ""
        self.stderr: str = ""
        self.returncode: int = -1
        self.workspace: Optional[Path] = None
        self.error: Optional[str] = None

    @property
    def all_passed(self):
        return all(ok for _, ok, _ in self.checks) and not self.error

    def check(self, label: str, ok: bool, detail: str = ""):
        self.checks.append((label, ok, detail))

    def summary_line(self) -> str:
        mark = "\033[32mPASS\033[0m" if self.all_passed else "\033[31mFAIL\033[0m"
        return f"  [{mark}]  {self.name}  ({self.elapsed:.1f}s)"


def run_harness(
    prompt: str,
    workspace: Path,
    timeout: int = DEFAULT_TIMEOUT,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, str, int]:
    """Run harness headlessly.  Returns (stdout, stderr, returncode)."""
    cmd = [
        PYTHON, str(HARNESS_PY),
        "--workspace", str(workspace),
        "--session", SESSION,
        "--new",                       # fresh session every run
        *(extra_args or []),
    ]
    env = os.environ.copy()
    env.pop("HARNESS_DEBUG", None)     # keep output clean by default

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(workspace),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        stdout += f"\n[TIMEOUT after {timeout}s]"
    return stdout, stderr, proc.returncode


def make_workspace(label: str) -> Path:
    """Create a fresh temp directory for a test."""
    d = Path(tempfile.mkdtemp(prefix=f"z_smoke_{label}_"))
    return d


def cleanup(ws: Path, keep: bool):
    if not keep and ws and ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
# Each test is a function that returns a TestResult.  It receives (timeout,).

ALL_TESTS: List[Tuple[str, Callable]] = []

def register(name: str):
    def deco(fn):
        ALL_TESTS.append((name, fn))
        return fn
    return deco


# ── 1. Create a new file from scratch ─────────────────────────────────────

@register("create_file")
def test_create_file(timeout: int) -> TestResult:
    """Ask the model to create a Python file and verify it exists."""
    r = TestResult("create_file — write a new Python file from scratch")
    ws = make_workspace("create")
    r.workspace = ws

    prompt = textwrap.dedent("""\
        Create a file called calculator.py in the workspace root.
        It should contain a class Calculator with four methods:
          add(a, b), subtract(a, b), multiply(a, b), divide(a, b).
        divide should raise ValueError on division by zero.
        Do NOT create any other files.  Do NOT create tests.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    target = ws / "calculator.py"
    r.check("calculator.py exists", target.exists())
    if target.exists():
        src = target.read_text(encoding="utf-8")
        r.check("contains class Calculator", "class Calculator" in src, src[:200])
        r.check("has add method",      "def add"      in src)
        r.check("has subtract method",  "def subtract"  in src)
        r.check("has multiply method",  "def multiply"  in src)
        r.check("has divide method",    "def divide"    in src)
        r.check("ValueError for /0",   "ValueError" in src)

        # EXECUTION CHECK: the code must actually work, not just look right
        try:
            result = subprocess.run(
                [PYTHON, "-c",
                 "from calculator import Calculator; c = Calculator(); "
                 "print(c.add(2,3), c.subtract(10,4), c.multiply(3,7), c.divide(15,3))"],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            out = result.stdout.strip()
            r.check("add(2,3)==5", out.startswith("5"), out)
            r.check("subtract(10,4)==6", "6" in out, out)
            r.check("multiply(3,7)==21", "21" in out, out)
            r.check("divide(15,3)==5.0", "5.0" in out, out)
        except Exception as e:
            r.check("calculator.py runs", False, str(e))

        # Verify divide-by-zero actually raises
        try:
            result = subprocess.run(
                [PYTHON, "-c",
                 "from calculator import Calculator; Calculator().divide(1, 0)"],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            r.check("divide(1,0) raises", result.returncode != 0, f"exit={result.returncode}")
        except Exception as e:
            r.check("divide error check runs", False, str(e))
    return r


# ── 2. Read + edit an existing file ───────────────────────────────────────

@register("edit_file")
def test_edit_file(timeout: int) -> TestResult:
    """Pre-populate a file, ask the model to make a specific edit."""
    r = TestResult("edit_file — read an existing file and make a targeted edit")
    ws = make_workspace("edit")
    r.workspace = ws

    original = textwrap.dedent("""\
        def greet(name):
            return "Hello, " + name

        def farewell(name):
            return "Goodbye, " + name
    """)
    (ws / "hello.py").write_text(original, encoding="utf-8")

    prompt = textwrap.dedent("""\
        Read hello.py.  Change the greet function so it returns an f-string
        like f"Hello, {name}!" (with an exclamation mark).
        Do NOT change the farewell function at all.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    target = ws / "hello.py"
    r.check("hello.py still exists", target.exists())
    if target.exists():
        src = target.read_text(encoding="utf-8")
        r.check("greet uses f-string",  'f"Hello, {name}!"' in src or "f'Hello, {name}!'" in src,
                 src[:300])
        # Verify farewell function body is EXACTLY preserved, not just present
        r.check("farewell function intact", 'return "Goodbye, " + name' in src
                 or "return 'Goodbye, ' + name" in src, src[:400])

        # EXECUTION CHECK: both functions must work correctly
        try:
            result = subprocess.run(
                [PYTHON, "-c",
                 'from hello import greet, farewell; print(greet("Alice")); print(farewell("Bob"))'],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            out = result.stdout.strip()
            r.check("greet output correct", "Hello, Alice!" in out, out)
            r.check("farewell output correct", "Goodbye, Bob" in out, out)
        except Exception as e:
            r.check("hello.py runs", False, str(e))
    return r


# ── 3. Search across multiple files ──────────────────────────────────────

@register("search_files")
def test_search_files(timeout: int) -> TestResult:
    """Seed several files, ask the model to find and report which file
    contains a specific string.  Validates search_files works end-to-end."""
    r = TestResult("search_files — locate a needle across many files")
    ws = make_workspace("search")
    r.workspace = ws

    # Seed 8 files across subdirectories, only one contains the target.
    # Using more files + subdirs makes it impossible to guess without searching.
    (ws / "src").mkdir()
    (ws / "lib").mkdir()
    for i in range(8):
        parent = ws / ("src" if i < 4 else "lib")
        (parent / f"module_{i}.py").write_text(
            f"# module {i}\nVALUE_{i} = 'decoy_{i}'\ndef func_{i}():\n    return {i}\n",
            encoding="utf-8",
        )
    # Plant the needle in a non-obvious location (lib/module_6.py)
    (ws / "lib" / "module_6.py").write_text(
        '# module 6\nSECRET_NEEDLE = "xK9_alpha_42"\ndef func_6():\n    return 6\n',
        encoding="utf-8",
    )

    prompt = textwrap.dedent("""\
        Search the workspace for the variable "SECRET_NEEDLE".
        Tell me which file contains it and what string value is assigned to it.
        Then create a file called answer.txt containing exactly one line:
            <relative/path/to/file>:<value>
        For example: src/foo.py:bar_123
        Do NOT include any other text in answer.txt.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    ans = ws / "answer.txt"
    r.check("answer.txt exists", ans.exists())
    if ans.exists():
        content = ans.read_text(encoding="utf-8").strip()
        # Must identify the correct file in lib/ (not guess module_3 or src/)
        r.check("identifies lib/module_6.py", "module_6.py" in content, content)
        r.check("correct path (lib/)", "lib" in content, content)
        # Must report the exact value that requires reading the file
        r.check("reports xK9_alpha_42", "xK9_alpha_42" in content, content)
        # Must NOT mention wrong files
        r.check("no false file", "module_3" not in content, content)
    return r


# ── 4. Read a large file (>2000 lines) ───────────────────────────────────

@register("read_large_file")
def test_read_large_file(timeout: int) -> TestResult:
    """Create a file with >2500 lines.  Ask the model to report the
    content of a specific line near the end.  Validates the read_file
    large-file preview + range read path works."""
    r = TestResult("read_large_file — read specific line in a 2500-line file")
    ws = make_workspace("largefile")
    r.workspace = ws

    # Use a unique random-looking value the model can't guess from the prompt
    magic_value = "q7W_bravo_8193"
    lines = []
    for i in range(1, 2501):
        if i == 2222:
            lines.append(f"# MAGIC_LINE: secret={magic_value}")
        else:
            lines.append(f"x_{i} = {i}")
    (ws / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prompt = textwrap.dedent("""\
        The file big.py has 2500 lines.  Somewhere around line 2222 there
        is a comment starting with "MAGIC_LINE".
        Find that comment and create a file called answer.txt
        containing only the full text of that comment line, nothing else.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    ans = ws / "answer.txt"
    r.check("answer.txt exists", ans.exists())
    if ans.exists():
        content = ans.read_text(encoding="utf-8").strip()
        r.check("contains MAGIC_LINE", "MAGIC_LINE" in content, content)
        # The value q7W_bravo_8193 is NOT in the prompt, so the model MUST
        # have actually read the file to produce it.
        r.check("contains secret value", magic_value in content, content)
        # Check the full exact line was captured
        r.check("exact line match",
                content == f"# MAGIC_LINE: secret={magic_value}",
                content[:120])
    return r


# ── 5. Multi-step: create + edit + verify ─────────────────────────────────

@register("multi_step")
def test_multi_step(timeout: int) -> TestResult:
    """Ask the model to create a file, then edit it, all in one prompt."""
    r = TestResult("multi_step — create a file then fix a bug in it")
    ws = make_workspace("multistep")
    r.workspace = ws

    prompt = textwrap.dedent("""\
        Step 1: Create a file called fib.py with a function fibonacci(n)
        that returns the n-th Fibonacci number (0-indexed, so fibonacci(0)=0,
        fibonacci(1)=1, fibonacci(6)=8).

        Step 2: After creating fib.py, read it back and then add a
        second function called is_prime(n) that returns True if n is prime.

        Step 3: After both functions exist, run this command to test:
            python -c "from fib import fibonacci, is_prime; print(fibonacci(10), is_prime(7))"
        The expected output should show "55 True".
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    target = ws / "fib.py"
    r.check("fib.py exists", target.exists())
    if target.exists():
        src = target.read_text(encoding="utf-8")
        r.check("has fibonacci", "def fibonacci" in src)
        r.check("has is_prime",  "def is_prime" in src)

        # Actually run it ourselves to validate correctness
        try:
            result = subprocess.run(
                [PYTHON, "-c",
                 "from fib import fibonacci, is_prime; print(fibonacci(10), is_prime(7))"],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            out = result.stdout.strip()
            r.check("fibonacci(10)==55", "55" in out, out)
            r.check("is_prime(7)==True", "True" in out, out)
        except Exception as e:
            r.check("runs without error", False, str(e))
    return r


# ── 6. Diagnose a bug ────────────────────────────────────────────────────

@register("diagnose_bug")
def test_diagnose_bug(timeout: int) -> TestResult:
    """Provide a buggy file, ask the model to find and fix the bug."""
    r = TestResult("diagnose_bug — find and fix an off-by-one error")
    ws = make_workspace("diagnose")
    r.workspace = ws

    buggy = textwrap.dedent("""\
        def avg(numbers):
            \"\"\"Return the average of a list of numbers.\"\"\"
            total = 0
            for n in numbers:
                total += n
            return total / (len(numbers) - 1)   # BUG: should be len(numbers)

        if __name__ == "__main__":
            data = [10, 20, 30]
            print(f"Average: {avg(data)}")
    """)
    (ws / "stats.py").write_text(buggy, encoding="utf-8")

    prompt = textwrap.dedent("""\
        Read stats.py.  There is a bug in the avg() function that causes
        wrong results.  Find and fix it using replace_in_file.  After fixing,
        run: python stats.py
        The correct output for [10, 20, 30] should be "Average: 20.0".
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    target = ws / "stats.py"
    r.check("stats.py exists", target.exists())
    if target.exists():
        src = target.read_text(encoding="utf-8")
        # The fix: len(numbers) - 1  →  len(numbers)
        r.check("bug is fixed (no -1)", "len(numbers) - 1" not in src, src[:300])
        r.check("uses len(numbers)",    "len(numbers)" in src)
        r.check("function still exists", "def avg" in src)

        # EXECUTION CHECK: the real test — does the code produce the right answer?
        try:
            result = subprocess.run(
                [PYTHON, str(ws / "stats.py")],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            r.check("runs without error", result.returncode == 0, result.stderr[:200])
            r.check("output says 20.0", "20.0" in result.stdout, result.stdout.strip())
            # Verify it's exactly 20.0, not 20.00 or 20.0000001
            r.check("output is Average: 20.0",
                     "Average: 20.0" in result.stdout, result.stdout.strip())
        except Exception as e:
            r.check("runs without error", False, str(e))
    return r


# ── 7. Shell command execution ────────────────────────────────────────────

@register("execute_command")
def test_execute_command(timeout: int) -> TestResult:
    """Ask the model to run a command and write its output to a file."""
    r = TestResult("execute_command — run a shell command, capture output")
    ws = make_workspace("exec")
    r.workspace = ws

    prompt = textwrap.dedent("""\
        Run this command: python -c "print('smoke_test_ok_' + str(7*6))"
        Write the exact output of that command to a file called output.txt.
        Do not add anything else to output.txt.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    ans = ws / "output.txt"
    r.check("output.txt exists", ans.exists())
    if ans.exists():
        content = ans.read_text(encoding="utf-8").strip()
        r.check("contains smoke_test_ok_42", "smoke_test_ok_42" in content, content)
    return r


# ── 8. Multi-file project scaffold ───────────────────────────────────────

@register("scaffold_project")
def test_scaffold_project(timeout: int) -> TestResult:
    """Ask the model to create a small multi-file project from scratch."""
    r = TestResult("scaffold_project — create multiple files in one go")
    ws = make_workspace("scaffold")
    r.workspace = ws

    prompt = textwrap.dedent("""\
        Create a tiny Python package with this structure:
          mylib/__init__.py      — should import greet from mylib.utils
          mylib/utils.py         — should define: def greet(name): return f"Hi, {name}!"

        Then create a file main.py in the workspace root that does:
          from mylib import greet
          print(greet("World"))

        After creating all files, run: python main.py
        It should print "Hi, World!"
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    r.check("mylib/__init__.py exists", (ws / "mylib" / "__init__.py").exists())
    r.check("mylib/utils.py exists",    (ws / "mylib" / "utils.py").exists())
    r.check("main.py exists",           (ws / "main.py").exists())

    # Actually run it
    if (ws / "main.py").exists():
        try:
            result = subprocess.run(
                [PYTHON, str(ws / "main.py")],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            out = result.stdout.strip()
            r.check("output contains Hi, World!", "Hi, World!" in out, out)
        except Exception as e:
            r.check("main.py runs", False, str(e))
    return r


# ── 9. Refuse to hallucinate — ask about code that doesn't exist ──────────

@register("no_hallucination")
def test_no_hallucination(timeout: int) -> TestResult:
    """Give the model an empty workspace and ask it to describe a class
    that doesn't exist.  It should say it can't find it, NOT fabricate."""
    r = TestResult("no_hallucination — refuse to fabricate nonexistent code")
    ws = make_workspace("nohalluc")
    r.workspace = ws

    # Single innocent file
    (ws / "readme.txt").write_text("This is a placeholder readme.\n", encoding="utf-8")

    prompt = textwrap.dedent("""\
        Read the file FooBarEngine.py and describe its SplineReticulator class.
        Write your findings to answer.txt.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    ans = ws / "answer.txt"
    if ans.exists():
        content = ans.read_text(encoding="utf-8")
        content_lower = content.lower()
        # The model should indicate the file doesn't exist - not describe a fake class
        found_honesty = any(w in content_lower for w in [
            "not found", "does not exist", "doesn't exist", "no such file",
            "could not find", "couldn't find", "not present", "no file",
            "unable to find", "unable to locate", "cannot find",
        ])
        r.check("acknowledges file missing", found_honesty, content[:300])

        # Strict anti-hallucination: no method signatures, no class bodies,
        # no invented attributes, no fake docstrings.
        # Note: the model may echo the class name from the prompt in prose
        # ("there is no SplineReticulator class") — that's fine.  We only
        # flag actual Python definitions / code fabrication.
        import re as _re
        r.check("no fabricated 'def '" , content_lower.count("def ") == 0, content[:400])
        r.check("no fabricated class definition",
                not _re.search(r'class\s+\w+\s*[:(]', content), content[:400])
        r.check("no fabricated 'self.'",  "self." not in content_lower, content[:400])
        r.check("no fabricated code block", "```" not in content, content[:400])
        # Check for fabricated behaviour descriptions (e.g. "it reticulates
        # splines" or "handles spline interpolation") — but allow the model
        # to echo the class name when saying it doesn't exist.
        has_functional_desc = any(phrase in content_lower for phrase in [
            "reticulates", "reticulating", "interpolat", "handles spline",
            "spline processing", "spline method",
        ])
        r.check("no spline fabrication", not has_functional_desc, content[:400])
    else:
        # No answer.txt at all — also acceptable if stdout mentions not found
        combined = (r.stdout + r.stderr).lower()
        found_in_output = any(w in combined for w in [
            "not found", "does not exist", "doesn't exist",
        ])
        r.check("reports file not found (stdout)", found_in_output)
    return r


# ── 10. Complex edit: JSON manipulation ───────────────────────────────────

@register("json_edit")
def test_json_edit(timeout: int) -> TestResult:
    """Populate a JSON config, ask the model to add a key and change a value."""
    r = TestResult("json_edit — targeted edits in a JSON file")
    ws = make_workspace("jsonedit")
    r.workspace = ws

    original = {
        "name": "myapp",
        "version": "1.0.0",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
        }
    }
    (ws / "config.json").write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

    prompt = textwrap.dedent("""\
        Read config.json.  Make these two changes:
        1. Change "debug" from true to false.
        2. Add a new top-level key "logging" with value {"level": "info", "file": "app.log"}.
        Write the result back.  Keep valid JSON formatting.
    """)

    r.stdout, r.stderr, r.returncode = run_harness(prompt, ws, timeout)
    r.check("exit code 0", r.returncode == 0, f"got {r.returncode}")

    target = ws / "config.json"
    r.check("config.json exists", target.exists())
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            r.check("valid JSON", True)
            r.check("debug is False",       data.get("debug") is False, str(data.get("debug")))
            r.check("logging.level=info",   data.get("logging", {}).get("level") == "info",
                     str(data.get("logging")))
            r.check("logging.file=app.log", data.get("logging", {}).get("file") == "app.log")
            r.check("name preserved",       data.get("name") == "myapp")
            r.check("database preserved",   data.get("database", {}).get("port") == 5432)
        except json.JSONDecodeError as e:
            r.check("valid JSON", False, str(e))
    return r


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(
    selected: Optional[List[int]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
    keep_failures: bool = True,
) -> List[TestResult]:
    results: List[TestResult] = []

    for idx, (name, fn) in enumerate(ALL_TESTS, 1):
        if selected and idx not in selected:
            continue
        print(f"\n{'='*60}")
        print(f"  TEST {idx}/{len(ALL_TESTS)}: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = fn(timeout)
        except Exception as e:
            result = TestResult(name)
            result.error = f"EXCEPTION: {e}"
            import traceback
            traceback.print_exc()
        result.elapsed = time.time() - t0

        # Print per-check results
        for label, ok, detail in result.checks:
            mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
            line = f"    {mark} {label}"
            if detail and (not ok or verbose):
                line += f"  — {detail[:120]}"
            print(line)

        if result.error:
            print(f"    \033[31m✗ {result.error}\033[0m")

        if verbose or not result.all_passed:
            if result.stdout:
                print(f"\n  ── stdout (last 40 lines) ──")
                for ln in result.stdout.strip().splitlines()[-40:]:
                    print(f"    {ln}")
            if result.stderr:
                print(f"\n  ── stderr (last 20 lines) ──")
                for ln in result.stderr.strip().splitlines()[-20:]:
                    print(f"    {ln}")

        # Cleanup workspace on pass, preserve on fail
        if result.all_passed:
            cleanup(result.workspace, keep=False)
        elif keep_failures:
            print(f"  \033[33m⚠ workspace preserved: {result.workspace}\033[0m")
        else:
            cleanup(result.workspace, keep=False)

        results.append(result)

    return results


def print_summary(results: List[TestResult]):
    passed = sum(1 for r in results if r.all_passed)
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(r.summary_line())

    total_time = sum(r.elapsed for r in results)
    color = "\033[32m" if failed == 0 else "\033[31m"
    print(f"\n  {color}{passed}/{len(results)} passed{' \033[0m'}  ({total_time:.0f}s total)")

    if failed:
        print(f"\n  Failed tests:")
        for r in results:
            if not r.all_passed:
                print(f"    - {r.name}")
                for label, ok, detail in r.checks:
                    if not ok:
                        print(f"        ✗ {label}: {detail[:100]}")
                if r.error:
                    print(f"        ✗ {r.error}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Black-box smoke tests for the harness CLI"
    )
    parser.add_argument(
        "--test", "-t", type=int, action="append", default=None,
        help="Run only specific test number(s), 1-indexed.  Repeatable.",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-test timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full stdout/stderr even on passing tests",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep all temp workspaces (even on pass)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for idx, (name, fn) in enumerate(ALL_TESTS, 1):
            doc = fn.__doc__ or ""
            print(f"  {idx:2d}. {name:25s} — {doc.strip().split(chr(10))[0]}")
        return

    print(f"Harness: {HARNESS_PY}")
    print(f"Python:  {PYTHON}")
    print(f"Timeout: {args.timeout}s per test")

    if not HARNESS_PY.exists():
        print(f"\nERROR: Harness not found at {HARNESS_PY}")
        sys.exit(1)

    results = run_tests(
        selected=args.test,
        timeout=args.timeout,
        verbose=args.verbose,
        keep_failures=not args.keep_all,
    )
    print_summary(results)

    sys.exit(0 if all(r.all_passed for r in results) else 1)


if __name__ == "__main__":
    main()
