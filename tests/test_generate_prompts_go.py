"""
Automated test: Use Python harness to generate prompts.go via MiniMax.

This test iterates until the harness can successfully:
1. Accept a prompt to write prompts.go
2. Execute write_to_file with XML documentation in the content
3. Produce valid Go code that compiles
"""

import subprocess
import sys
import os
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).parent.parent
HARNESS_GO_DIR = HARNESS_DIR / "harness_go"
PROMPTS_GO = HARNESS_GO_DIR / "prompts.go"

# Simple prompt that asks for a minimal but valid prompts.go
PROMPT = '''Write a complete prompts.go file for harness_go/ that includes:

1. A getSystemPrompt(workspace string) function that:
   - Checks for .harness_prompt file override
   - Returns defaultSystemPrompt with %%OS%%, %%SHELL%%, %%WORKSPACE%% replaced

2. An init() function that detects runtime.GOOS and sets os/shell variables

3. A defaultSystemPrompt constant (use backticks) with documentation for these tools:
   - read_file: read a file, params: path
   - write_to_file: create new file, params: path, content
   - replace_in_file: edit file with SEARCH/REPLACE blocks, params: path, diff
   - execute_command: run shell command, params: command, background
   - list_files: list directory, params: path, recursive
   - search_files: regex search, params: path, regex, file_pattern

Include XML examples for each tool in the documentation.

Use write_to_file to create the complete file. The file must compile with `go build`.
'''


def get_original_content():
    """Get the minimal prompts.go content."""
    return '''package main

import (
	"os"
	"path/filepath"
	"strings"
)

const defaultSystemPrompt = `TODO: Add full prompt`

func getSystemPrompt(workspace string) string {
	promptPath := filepath.Join(workspace, ".harness_prompt")
	if _, err := os.Stat(promptPath); err == nil {
		if content, err := os.ReadFile(promptPath); err == nil {
			return strings.TrimSpace(string(content))
		}
	}
	return defaultSystemPrompt
}
'''


def reset_prompts_go():
    """Reset prompts.go to minimal state."""
    PROMPTS_GO.write_text(get_original_content())
    print(f"Reset {PROMPTS_GO}")


def run_harness(prompt: str, timeout: int = 180) -> tuple[bool, str]:
    """Run harness with piped input. Returns (success, output)."""
    try:
        # Use Popen with binary mode to avoid encoding issues
        env = {**os.environ, "HARNESS_DEBUG": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "harness", str(HARNESS_GO_DIR), "--new"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(HARNESS_DIR),
            env=env,
        )
        stdout_bytes, _ = proc.communicate(input=prompt.encode('utf-8'), timeout=timeout)
        output = stdout_bytes.decode('utf-8', errors='replace')
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_go_compiles() -> tuple[bool, str]:
    """Check if Go harness compiles."""
    try:
        result = subprocess.run(
            ["go", "build", "."],
            capture_output=True,
            text=True,
            cwd=str(HARNESS_GO_DIR),
            timeout=30
        )
        if result.returncode == 0:
            return True, "Compiles OK"
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def check_prompts_go_updated() -> bool:
    """Check if prompts.go has meaningful content."""
    if not PROMPTS_GO.exists():
        return False
    content = PROMPTS_GO.read_text()
    # Must have more than minimal content and include key components
    return (
        len(content) > 500 and
        "init()" in content and
        "runtime" in content and
        "read_file" in content.lower()
    )


def run_test(max_attempts: int = 1):
    """Run the generation test with retries."""
    print("=" * 60)
    print("Testing: Generate prompts.go via Python harness")
    print("=" * 60)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        # Reset to known state
        reset_prompts_go()
        initial_size = PROMPTS_GO.stat().st_size
        
        # Run harness
        print("Running harness...")
        success, output = run_harness(PROMPT)
        
        # Save output for analysis
        output_file = HARNESS_DIR / f"test_output_{attempt}.txt"
        output_file.write_text(output, encoding='utf-8', errors='replace')
        print(f"Output saved to: {output_file}")
        
        if not success:
            print(f"Harness failed: {output[:500]}")
            continue
        
        # Check if file was updated
        if not check_prompts_go_updated():
            current_size = PROMPTS_GO.stat().st_size
            print(f"prompts.go not updated (was {initial_size}b, now {current_size}b)")
            print("Last output snippet:")
            print(output[-1000:] if len(output) > 1000 else output)
            continue
        
        # Check if it compiles
        compiles, msg = check_go_compiles()
        if not compiles:
            print(f"Go build failed: {msg}")
            continue
        
        # Success!
        final_size = PROMPTS_GO.stat().st_size
        print(f"\n✓ SUCCESS! prompts.go is {final_size} bytes and compiles.")
        print(f"Content preview:")
        content = PROMPTS_GO.read_text()
        print(content[:500] + "..." if len(content) > 500 else content)
        return True
    
    print(f"\n✗ FAILED after {max_attempts} attempts")
    return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
