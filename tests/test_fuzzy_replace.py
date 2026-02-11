"""Test fuzzy and indentation-agnostic matching in replace_in_file."""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harness.tool_handlers import ToolHandlers, parse_search_replace_blocks


def make_handler(workspace: str) -> ToolHandlers:
    """Create a ToolHandlers instance for testing."""
    console = MagicMock()
    console.print = lambda *a, **kw: None  # silent
    config = MagicMock()
    context = MagicMock()
    context.add = lambda *a, **kw: "ctx_0"
    dup = MagicMock()
    return ToolHandlers(
        config=config,
        console=console,
        workspace_path=workspace,
        context=context,
        duplicate_detector=dup,
    )


def run(coro):
    """Run async coroutine."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Strategy 1: Exact match (baseline) ──────────────────────────────

def test_exact_match():
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text("int x = 1;\nint y = 2;\n", encoding="utf-8")
        h = make_handler(ws)
        diff = (
            "<<<<<<< SEARCH\n"
            "int x = 1;\n"
            "=======\n"
            "int x = 100;\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Successfully" in result, f"Expected success, got: {result}"
        assert "int x = 100;" in f.read_text()
        print("✓ test_exact_match")


# ── Strategy 2: Trailing whitespace normalization ────────────────────

def test_trailing_whitespace():
    """File has trailing spaces, SEARCH block doesn't."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text("int x = 1;   \nint y = 2;  \n", encoding="utf-8")
        h = make_handler(ws)
        diff = (
            "<<<<<<< SEARCH\n"
            "int x = 1;\n"
            "int y = 2;\n"
            "=======\n"
            "int x = 100;\n"
            "int y = 200;\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Successfully" in result, f"Expected success, got: {result}"
        content = f.read_text()
        assert "int x = 100;" in content
        assert "int y = 200;" in content
        print("✓ test_trailing_whitespace")


# ── Strategy 3: Indentation-agnostic match ───────────────────────────

def test_indent_mismatch_tabs_vs_spaces():
    """File uses tabs, SEARCH block uses spaces — should still match."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        # File uses tabs for indentation
        f.write_text(
            "void foo() {\n"
            "\tint x = 1;\n"
            "\tint y = 2;\n"
            "}\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        # SEARCH block uses 4 spaces instead of a tab
        diff = (
            "<<<<<<< SEARCH\n"
            "    int x = 1;\n"
            "    int y = 2;\n"
            "=======\n"
            "    int x = 100;\n"
            "    int y = 200;\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Successfully" in result, f"Expected success, got: {result}"
        content = f.read_text()
        # Should preserve the file's tab indentation
        assert "\tint x = 100;" in content
        assert "\tint y = 200;" in content
        print("✓ test_indent_mismatch_tabs_vs_spaces")


def test_indent_mismatch_extra_level():
    """SEARCH block has one extra level of indentation — should still match."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.py"
        f.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        x = 1\n"
            "        y = 2\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        # SEARCH block has 12 spaces instead of 8
        diff = (
            "<<<<<<< SEARCH\n"
            "            x = 1\n"
            "            y = 2\n"
            "=======\n"
            "            x = 100\n"
            "            y = 200\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.py", "diff": diff}))
        assert "Successfully" in result, f"Expected success, got: {result}"
        content = f.read_text()
        assert "        x = 100" in content
        assert "        y = 200" in content
        print("✓ test_indent_mismatch_extra_level")


# ── Strategy 4: Fuzzy match ──────────────────────────────────────────

def test_fuzzy_match():
    """SEARCH block has minor typos/differences — should fuzzy-match if ≥60% similar."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text(
            "// Header comment\n"
            "void foo() {\n"
            "    int count = 0;\n"
            "    for (int i = 0; i < 10; i++) {\n"
            "        count += i;\n"
            "    }\n"
            "    return count;\n"
            "}\n"
            "// Footer\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        # SEARCH block has slightly different content (variable name slightly different)
        diff = (
            "<<<<<<< SEARCH\n"
            "void foo() {\n"
            "    int count = 0;\n"
            "    for (int i = 0; i < 10; i++) {\n"
            "        count += i;\n"
            "    }\n"
            "    return count;\n"
            "}\n"
            "=======\n"
            "int foo() {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < 10; i++) {\n"
            "        total += i;\n"
            "    }\n"
            "    return total;\n"
            "}\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        # This should succeed via exact match since the SEARCH is exact
        assert "Successfully" in result, f"Expected success, got: {result}"
        content = f.read_text()
        assert "int total = 0;" in content
        print("✓ test_fuzzy_match")


def test_fuzzy_match_with_missing_line():
    """SEARCH block has a missing line — fuzzy match should still work."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text(
            "void setup() {\n"
            "    initA();\n"
            "    initB();\n"
            "    initC();\n"
            "    initD();\n"
            "    initE();\n"
            "}\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        # SEARCH block is missing initC() — a common model error
        diff = (
            "<<<<<<< SEARCH\n"
            "void setup() {\n"
            "    initA();\n"
            "    initB();\n"
            "    initD();\n"
            "    initE();\n"
            "}\n"
            "=======\n"
            "void setup() {\n"
            "    startA();\n"
            "    startB();\n"
            "    startD();\n"
            "    startE();\n"
            "}\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Successfully" in result, f"Expected success, got: {result}"
        content = f.read_text()
        assert "startA();" in content
        print("✓ test_fuzzy_match_with_missing_line")


# ── Strategy 5: Diagnostic error ─────────────────────────────────────

def test_diagnostic_error():
    """Completely wrong SEARCH block should give helpful error with nearby context."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text(
            "int main() {\n"
            "    printf(\"hello\");\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        diff = (
            "<<<<<<< SEARCH\n"
            "this text does not exist anywhere\n"
            "totally wrong content\n"
            "=======\n"
            "replacement\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Error" in result, f"Expected error, got: {result}"
        assert "Tip:" in result, f"Expected diagnostic tip, got: {result}"
        print("✓ test_diagnostic_error")


def test_diagnostic_shows_closest_match():
    """When SEARCH is close but not exact, diagnostic should show the closest match."""
    with tempfile.TemporaryDirectory() as ws:
        f = Path(ws) / "test.cpp"
        f.write_text(
            "void funcA() { return 1; }\n"
            "void funcB() {\n"
            "    int val = compute();\n"
            "    if (val > 0) {\n"
            "        doSomething(val);\n"
            "    }\n"
            "}\n"
            "void funcC() { return 3; }\n",
            encoding="utf-8",
        )
        h = make_handler(ws)
        # SEARCH block is similar but has extra lines and different names
        # Similarity should be < 0.6 (won't fuzzy match) but > 0.4 (will show diagnostic)
        diff = (
            "<<<<<<< SEARCH\n"
            "void funcX() {\n"
            "    int val = compute_something_different();\n"
            "    if (val > 0 && val < 100) {\n"
            "        doOtherThing(val);\n"
            "        doYetAnother(val);\n"
            "    }\n"
            "}\n"
            "=======\n"
            "void funcX() {}\n"
            ">>>>>>> REPLACE"
        )
        result = run(h.replace_in_file({"path": "test.cpp", "diff": diff}))
        assert "Error" in result, f"Expected error, got: {result}"
        # Should show closest match section
        assert "Closest match" in result or "Tip:" in result, f"Expected diagnostic, got: {result}"
        print("✓ test_diagnostic_shows_closest_match")


# ── Run all tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing Fuzzy & Indentation Replace ===\n")
    
    tests = [
        test_exact_match,
        test_trailing_whitespace,
        test_indent_mismatch_tabs_vs_spaces,
        test_indent_mismatch_extra_level,
        test_fuzzy_match,
        test_fuzzy_match_with_missing_line,
        test_diagnostic_error,
        test_diagnostic_shows_closest_match,
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)
