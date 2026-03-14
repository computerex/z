"""Headless test: verify the model can see and replace </think> tokens in files.

This test:
1. Creates a test file containing a literal </think> token
2. Asks the harness (headlessly via stdin) to find and replace it
3. Verifies the replacement was made

Requires a configured harness with API access (reads ~/.z.json).
"""

import subprocess
import sys
import os
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
TEST_FILE = WORKSPACE / "think_token_test.txt"

ORIGINAL_CONTENT = """\
This is a test file for think token handling.
Line 2: normal text here.
Line 3: </think>
Line 4: more text after the token.
Line 5: end of file.
"""

PROMPT = (
    "Read the file think_token_test.txt in the current workspace directory. "
    "Line 3 contains a special XML-like closing tag token that needs to be replaced. "
    "Use the replace_in_file tool to find that tag on line 3 and replace it with the word REPLACED. "
    "Do NOT overwrite the entire file. You MUST use replace_in_file."
)


def reset_test_file():
    """Reset the test file to its original content."""
    TEST_FILE.write_text(ORIGINAL_CONTENT, encoding="utf-8")
    print(f"[setup] Wrote test file: {TEST_FILE}")
    print(f"[setup] Content:\n{ORIGINAL_CONTENT}")


def run_harness(prompt: str, timeout: int = 120) -> tuple:
    """Run the harness headlessly and return (stdout, stderr, returncode)."""
    env = os.environ.copy()
    env["HARNESS_DEBUG"] = "1"

    cmd = [sys.executable, str(WORKSPACE / "harness.py"), "--workspace", str(WORKSPACE)]

    print(f"\n[run] Executing: {' '.join(cmd)}")
    print(f"[run] Prompt: {prompt[:200]}...")
    print(f"[run] Timeout: {timeout}s")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(WORKSPACE),
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
        print(f"[run] TIMEOUT after {timeout}s")

    return stdout, stderr, proc.returncode


def check_result() -> bool:
    """Check if the test file was correctly modified."""
    if not TEST_FILE.exists():
        print("[check] FAIL: test file does not exist!")
        return False

    content = TEST_FILE.read_text(encoding="utf-8")
    print(f"\n[check] File content after harness run:\n{content}")

    # Check that </think> was replaced
    if "</think>" in content:
        print("[check] FAIL: </think> still present in file")
        return False

    if "REPLACED" in content:
        print("[check] PASS: </think> was successfully replaced with REPLACED")
        return True

    print("[check] FAIL: REPLACED not found in file (unexpected modification)")
    return False


def main():
    print("=" * 60)
    print("Think Token Headless Test")
    print("=" * 60)

    # Reset test file
    reset_test_file()

    # Run harness
    start = time.time()
    stdout, stderr, retcode = run_harness(PROMPT)
    elapsed = time.time() - start

    print(f"\n[run] Completed in {elapsed:.1f}s (exit code: {retcode})")

    # Show output (truncated)
    if stdout:
        lines = stdout.strip().split("\n")
        print(f"\n[stdout] ({len(lines)} lines):")
        for line in lines[-50:]:  # Show last 50 lines
            print(f"  {line}")

    if stderr:
        err_lines = stderr.strip().split("\n")
        # Show only non-debug stderr lines
        important = [l for l in err_lines if not l.startswith("DEBUG")]
        if important:
            print(f"\n[stderr] ({len(important)} important lines):")
            for line in important[-20:]:
                print(f"  {line}")

    # Check result
    print("\n" + "=" * 60)
    success = check_result()
    print("=" * 60)

    if success:
        print("\n>>> TEST PASSED <<<")
    else:
        print("\n>>> TEST FAILED <<<")
        print("The model could not find and replace </think> in the file.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
