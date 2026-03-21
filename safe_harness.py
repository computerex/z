#!/usr/bin/env python3
"""
SAFE MODE HARNESS - Minimal, bulletproof harness for self-repair.

This file is FROZEN and should NEVER be edited by the main harness.
It exists solely to fix the main harness when it breaks.

Features:
- Single file, ~300 lines, no internal dependencies
- Only requires: Python 3.8+, httpx (or falls back to urllib)
- Tools: read_file, write_file, replace_in_file, execute_command, attempt_completion
- Hardcoded to use GitHub Copilot (most reliable) or falls back to config

Usage:
    python safe_harness.py                    # Interactive mode
    python safe_harness.py --test             # Test if main harness works
    python safe_harness.py --fix              # Auto-fix mode (reads error, attempts fix)
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Edit these if needed
# ══════════════════════════════════════════════════════════════════════════════

HARNESS_DIR = Path(__file__).parent.absolute()
CONFIG_PATH = Path.home() / ".z.json"

# Default provider (GitHub Copilot is most reliable)
DEFAULT_API_URL = "https://api.githubcopilot.com/"
DEFAULT_MODEL = "claude-sonnet-4"

# ══════════════════════════════════════════════════════════════════════════════
# HTTP CLIENT - Try httpx, fall back to urllib
# ══════════════════════════════════════════════════════════════════════════════

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import urllib.request
    import urllib.error

def http_post(url: str, headers: dict, json_data: dict, timeout: int = 120) -> dict:
    """POST JSON and return response JSON."""
    if HAS_HTTPX:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=json_data)
            resp.raise_for_status()
            return resp.json()
    else:
        data = json.dumps(json_data).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> Dict[str, Any]:
    """Load config from ~/.z.json or use defaults."""
    config = {
        "api_url": DEFAULT_API_URL,
        "api_key": "",
        "model": DEFAULT_MODEL,
    }
    
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            # Prefer github-copilot provider if available
            providers = data.get("providers", {})
            if "github-copilot" in providers:
                p = providers["github-copilot"]
                config["api_url"] = p.get("api_url", config["api_url"])
                config["api_key"] = p.get("api_key", "")
                config["model"] = p.get("model", config["model"])
            else:
                config["api_url"] = data.get("api_url", config["api_url"])
                config["api_key"] = data.get("api_key", "")
                config["model"] = data.get("model", config["model"])
        except Exception as e:
            print(f"[WARN] Could not load config: {e}")
    
    return config

# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def tool_read_file(path: str) -> str:
    """Read a file and return its contents."""
    full_path = HARNESS_DIR / path
    if not full_path.exists():
        return f"Error: File not found: {path}"
    try:
        return full_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return f"Error reading file: {e}"

def tool_write_file(path: str, content: str) -> str:
    """Write content to a file."""
    full_path = HARNESS_DIR / path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def tool_replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Replace text in a file."""
    full_path = HARNESS_DIR / path
    if not full_path.exists():
        return f"Error: File not found: {path}"
    try:
        content = full_path.read_text(encoding='utf-8')
        if old_text not in content:
            # Show context to help debug
            return f"Error: old_text not found in {path}. File has {len(content)} chars."
        new_content = content.replace(old_text, new_text, 1)
        full_path.write_text(new_content, encoding='utf-8')
        return f"Successfully replaced text in {path}"
    except Exception as e:
        return f"Error: {e}"

def tool_execute_command(command: str) -> str:
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(HARNESS_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 300 seconds"
    except Exception as e:
        return f"Error executing command: {e}"

def tool_list_files(path: str, recursive: bool = False) -> str:
    """List files in a directory."""
    full_path = HARNESS_DIR / path
    if not full_path.exists():
        return f"Error: Directory not found: {path}"
    try:
        if recursive:
            files = [str(p.relative_to(full_path)) for p in full_path.rglob('*') if p.is_file()]
        else:
            files = [p.name for p in full_path.iterdir()]
        return '\n'.join(sorted(files)[:500])
    except Exception as e:
        return f"Error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# TOOL PARSER & EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

TOOL_PATTERN = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
PARAM_PATTERN = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from assistant response."""
    tools = []
    for match in TOOL_PATTERN.finditer(text):
        tool_name = match.group(1)
        tool_body = match.group(2)
        
        # Skip if it's a thinking block
        if tool_name in ('thinking', 'result'):
            continue
            
        params = {}
        for param_match in PARAM_PATTERN.finditer(tool_body):
            params[param_match.group(1)] = param_match.group(2).strip()
        
        tools.append({"name": tool_name, "params": params, "raw": match.group(0)})
    
    return tools

def execute_tool(name: str, params: Dict[str, str]) -> str:
    """Execute a tool and return the result."""
    if name == "read_file":
        return tool_read_file(params.get("path", ""))
    elif name == "write_to_file":
        return tool_write_file(params.get("path", ""), params.get("content", ""))
    elif name == "replace_in_file":
        return tool_replace_in_file(
            params.get("path", ""),
            params.get("old_text", ""),
            params.get("new_text", ""),
        )
    elif name == "execute_command":
        return tool_execute_command(params.get("command", ""))
    elif name == "list_files":
        return tool_list_files(
            params.get("path", "."),
            params.get("recursive", "").lower() == "true",
        )
    elif name == "attempt_completion":
        return "__COMPLETION__:" + params.get("result", "")
    else:
        return f"Unknown tool: {name}"

# ══════════════════════════════════════════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a harness repair assistant. Your job is to fix the main harness when it breaks.

You have these tools:
- read_file: Read a file. Params: path
- write_to_file: Write a file. Params: path, content
- replace_in_file: Replace text in a file. Params: path, old_text, new_text
- execute_command: Run a shell command. Params: command
- list_files: List directory contents. Params: path, recursive (optional)
- attempt_completion: Signal task completion. Params: result

Tool format:
<tool_name>
<param>value</param>
</tool_name>

Working directory: {cwd}

IMPORTANT:
1. First understand the error by reading files and running commands
2. Make minimal, targeted fixes
3. Test your fix with: python harness.py --help
4. Run tests with: pytest tests/ -x
5. Use attempt_completion when done
"""

def call_llm(messages: List[Dict], config: Dict) -> str:
    """Call the LLM and return the response text."""
    api_url = config["api_url"].rstrip('/')
    
    # Build request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    
    # Handle OAuth tokens
    if config["api_key"].startswith("oauth:"):
        headers["Authorization"] = f"Bearer {config['api_key'][6:]}"
    
    payload = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.3,
    }
    
    url = f"{api_url}/chat/completions"
    
    try:
        resp = http_post(url, headers, payload, timeout=120)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM Error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def test_main_harness() -> tuple[bool, str]:
    """Test if the main harness works. Returns (success, output)."""
    # Test 1: CLI help test (most basic - does it start?)
    result = tool_execute_command("python harness.py --help")
    if "Traceback" in result or "Error" in result.split('\n')[0]:
        return False, f"CLI test failed:\n{result}"
    
    # Test 2: Version/basic run test
    result = tool_execute_command("python harness.py --version 2>&1")
    if "Traceback" in result:
        return False, f"Version test failed:\n{result}"
    
    # Test 3: Import the harness modules directly
    result = tool_execute_command('python -c "import sys; sys.path.insert(0, \'src\'); from harness.cline_agent import ClineAgent; print(\'OK\')"')
    if "OK" not in result:
        return False, f"Module import test failed:\n{result}"
    
    return True, "All tests passed! Main harness is working."

def print_colored(text: str, color: str = ""):
    """Print with ANSI colors if supported."""
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m", "reset": "\033[0m"}
    if sys.stdout.isatty() and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)

def main():
    print_colored("=" * 60, "blue")
    print_colored("  SAFE MODE HARNESS - For repairing the main harness", "blue")
    print_colored("=" * 60, "blue")
    print()
    
    # Handle CLI args
    if "--test" in sys.argv:
        print("Testing main harness...")
        success, output = test_main_harness()
        print(output)
        sys.exit(0 if success else 1)
    
    if "--fix" in sys.argv:
        print("Auto-fix mode: Testing harness first...")
        success, output = test_main_harness()
        if success:
            print_colored("Main harness is working! No fix needed.", "green")
            sys.exit(0)
        print_colored("Main harness is broken. Starting repair...", "yellow")
        initial_prompt = f"The main harness is broken. Here's the error:\n\n{output}\n\nPlease diagnose and fix it."
    else:
        initial_prompt = None
    
    # Load config
    config = load_config()
    if not config["api_key"]:
        print_colored("Error: No API key found in ~/.z.json", "red")
        print("Please ensure you have a valid configuration.")
        sys.exit(1)
    
    print(f"Using: {config['api_url']} / {config['model']}")
    print(f"Working directory: {HARNESS_DIR}")
    print()
    print("Commands: /test (test harness), /quit (exit)")
    print("-" * 60)
    
    # Initialize conversation
    system = SYSTEM_PROMPT.format(cwd=HARNESS_DIR)
    messages = [{"role": "system", "content": system}]
    
    # Auto-start with fix prompt if in fix mode
    if initial_prompt:
        print(f"\n[Auto] {initial_prompt[:100]}...")
        messages.append({"role": "user", "content": initial_prompt})
    
    while True:
        # Get user input (unless we have an auto-prompt)
        if not initial_prompt:
            try:
                user_input = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting safe mode.")
                break
            
            if not user_input:
                continue
            if user_input == "/quit":
                break
            if user_input == "/test":
                success, output = test_main_harness()
                print(output)
                continue
            
            messages.append({"role": "user", "content": user_input})
        else:
            initial_prompt = None  # Clear so we prompt next iteration
        
        # Call LLM
        print("\n[Assistant] ", end="", flush=True)
        response = call_llm(messages, config)
        
        if response.startswith("LLM Error"):
            print_colored(response, "red")
            messages.pop()  # Remove failed user message
            continue
        
        # Display response (without tool XML for cleaner output)
        display_text = re.sub(r'<(\w+)>.*?</\1>', '[tool call]', response, flags=re.DOTALL)
        print(display_text.strip())
        
        messages.append({"role": "assistant", "content": response})
        
        # Execute any tool calls
        tools = extract_tool_calls(response)
        for tool in tools:
            print(f"\n[Executing: {tool['name']}]")
            result = execute_tool(tool["name"], tool["params"])
            
            if result.startswith("__COMPLETION__:"):
                print_colored(f"\n✓ Task completed: {result[15:]}", "green")
                # Don't add completion as a message, just show it
                continue
            
            print(f"[Result] {result[:500]}{'...' if len(result) > 500 else ''}")
            messages.append({"role": "user", "content": f"[{tool['name']} result]\n{result}"})
        
        # If there were tool calls, get next response automatically
        if tools and not any(t["name"] == "attempt_completion" for t in tools):
            continue  # Loop will call LLM again with tool results

if __name__ == "__main__":
    main()