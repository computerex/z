"""Behavioral E2E driver for native tool calling.

Drives a real ClineAgent against the configured live provider and captures
the rendered console output so we can assert on what the user actually sees.

Run:  python tests/playground/e2e_native.py [scenario]
"""

import os
import sys
import io
import asyncio
import tempfile
import shutil
from pathlib import Path

# Match the real `z` entry point: force UTF-8 stdio before importing harness.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", write_through=True)
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make src importable when run from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console

from harness.config import Config
from harness.cline_agent import ClineAgent


class _Tee:
    """Tee sys.stdout into a buffer so we capture live-streamed content too.

    In non-tty mode the agent streams visible content directly to sys.stdout
    (bypassing the rich Console). We capture both so assertions are robust
    regardless of whether output went via the Panel or the raw stream.
    """

    def __init__(self, real):
        self._real = real
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return self._real.isatty()

    def __getattr__(self, name):
        return getattr(self._real, name)


def make_agent(workspace: Path, record: bool = True):
    """Build an agent rooted at *workspace* with a recording console."""
    os.chdir(workspace)
    cfg = Config.from_json(workspace=workspace)
    # Recording console: capture everything printed for assertions.
    console = Console(record=record, force_terminal=False, width=100)
    agent = ClineAgent(config=cfg, console=console)
    return agent, console


async def run_turn(agent, console, text: str) -> str:
    print(f"\n{'=' * 70}\nUSER: {text}\n{'=' * 70}")
    tee = _Tee(sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        await agent.run_message(text, enable_interrupt=False)
    finally:
        sys.stdout = old
    # Combine both capture paths: rich Console export + raw stdout stream.
    return console.export_text() + "\n" + tee.buf.getvalue()


async def scenario_hi():
    """Reproduce the empty-box bug: a trivial greeting."""
    ws = Path(tempfile.mkdtemp(prefix="e2e_hi_"))
    try:
        agent, console = make_agent(ws)
        rendered = await run_turn(agent, console, "hi")
        print("\n--- RENDERED CONSOLE ---")
        print(rendered)
        print("--- END ---")
        # Dump message history for diagnosis
        print("\n--- MESSAGE HISTORY ---")
        for i, m in enumerate(agent.messages):
            c = m.content if isinstance(m.content, str) else str(m.content)
            tc = getattr(m, "tool_calls", None)
            print(f"[{i}] role={m.role} len={len(c or '')} tool_calls={bool(tc)}")
            if m.role in ("assistant", "tool") and c:
                print(f"      content[:160]={c[:160]!r}")
            if tc:
                for t in tc:
                    fn = t.get("function", {}) if isinstance(t, dict) else {}
                    print(f"      -> call {fn.get('name')!r} args={str(fn.get('arguments'))[:120]!r}")
        # The visible reply must be non-empty.
        visible = rendered.strip()
        assert visible, "FAIL: nothing rendered to console for 'hi'"
        # The last assistant message in history should have content.
        last = agent.messages[-1]
        print(f"\nlast msg role={last.role!r} content_len={len(last.content or '')}")
        assert last.role == "assistant", f"expected assistant last, got {last.role}"
        print("PASS: greeting rendered visible output")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def scenario_write_read():
    """Tool e2e: ask the agent to create a file and read it back."""
    ws = Path(tempfile.mkdtemp(prefix="e2e_wr_"))
    try:
        agent, console = make_agent(ws)
        rendered = await run_turn(
            agent,
            console,
            "Create a file called hello.txt containing exactly the text 'Hello E2E'. "
            "Then read it back to confirm. Use your tools.",
        )
        print("\n--- RENDERED CONSOLE ---")
        print(rendered)
        print("--- END ---")
        target = ws / "hello.txt"
        assert target.exists(), f"FAIL: {target} was not created"
        content = target.read_text(encoding="utf-8")
        print(f"\nfile content: {content!r}")
        assert "Hello E2E" in content, f"FAIL: unexpected content {content!r}"
        print("PASS: write+read tools worked")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def scenario_edit():
    """Tool e2e: edit an existing file with replace_in_file."""
    ws = Path(tempfile.mkdtemp(prefix="e2e_edit_"))
    try:
        (ws / "data.py").write_text("VALUE = 1\nNAME = 'old'\n", encoding="utf-8")
        agent, console = make_agent(ws)
        rendered = await run_turn(
            agent,
            console,
            "In data.py change NAME from 'old' to 'new'. Only change that line.",
        )
        print("\n--- RENDERED CONSOLE ---")
        print(rendered)
        print("--- END ---")
        content = (ws / "data.py").read_text(encoding="utf-8")
        print(f"\nfile content:\n{content}")
        assert "'new'" in content, f"FAIL: edit not applied: {content!r}"
        assert "VALUE = 1" in content, "FAIL: clobbered unrelated line"
        print("PASS: edit tool worked")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def scenario_search():
    """Tool e2e: search_files + list_files on a small tree."""
    ws = Path(tempfile.mkdtemp(prefix="e2e_search_"))
    try:
        (ws / "a.py").write_text("def alpha():\n    return MAGIC_TOKEN\n", encoding="utf-8")
        (ws / "b.py").write_text("def beta():\n    return 0\n", encoding="utf-8")
        (ws / "sub").mkdir()
        (ws / "sub" / "c.py").write_text("x = MAGIC_TOKEN + 1\n", encoding="utf-8")
        agent, console = make_agent(ws)
        rendered = await run_turn(
            agent,
            console,
            "Search the codebase for the string MAGIC_TOKEN and tell me exactly which "
            "files contain it. Use your search tool.",
        )
        print("\n--- RENDERED CONSOLE ---")
        print(rendered)
        print("--- END ---")
        low = rendered.lower()
        assert "a.py" in low and "c.py" in low, f"FAIL: did not report both files. Got:\n{rendered}"
        print("PASS: search tool worked")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


async def scenario_multistep():
    """Multi-step task exercising todos + command + file creation."""
    ws = Path(tempfile.mkdtemp(prefix="e2e_multi_"))
    try:
        agent, console = make_agent(ws)
        rendered = await run_turn(
            agent,
            console,
            "Create a Python script calc.py with a function add(a, b) that returns a+b, "
            "then run it with python to print add(2, 3). Confirm the output is 5.",
        )
        print("\n--- RENDERED CONSOLE ---")
        print(rendered)
        print("--- END ---")
        assert (ws / "calc.py").exists(), "FAIL: calc.py not created"
        src = (ws / "calc.py").read_text(encoding="utf-8")
        assert "def add" in src, f"FAIL: add() missing:\n{src}"
        assert "5" in rendered, "FAIL: output 5 not shown in transcript"
        print("PASS: multi-step task worked")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


SCENARIOS = {
    "hi": scenario_hi,
    "write_read": scenario_write_read,
    "edit": scenario_edit,
    "search": scenario_search,
    "multistep": scenario_multistep,
}


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for name, fn in SCENARIOS.items():
            print(f"\n\n########## SCENARIO: {name} ##########")
            await fn()
    else:
        await SCENARIOS[which]()


if __name__ == "__main__":
    asyncio.run(main())
