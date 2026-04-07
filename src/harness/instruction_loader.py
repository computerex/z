"""CLAUDE.md inheritance loader — mirrors Claude Code's memory hierarchy.

Resolution order (loaded bottom-up, more specific wins):
1. User-level:   ~/.claude/CLAUDE.md, ~/.claude/rules/*.md
2. Ancestor dirs: walk from filesystem root → cwd, at each level:
      <dir>/CLAUDE.md  or  <dir>/.claude/CLAUDE.md
3. Project rules: <workspace>/.claude/rules/*.md  (recursive)
4. agent.md:      <workspace>/agent.md  (harness-specific extension)

Features:
- @path imports (max depth 5, relative to the referencing file)
- HTML block-comment stripping (<!-- … -->)
- On-demand subdirectory loading (called when agent reads files in a subdir)
- Per-file 32 KB cap, total 128 KB cap
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Optional, Set

log = logging.getLogger("harness.instructions")

_MAX_FILE_BYTES = 32 * 1024  # 32 KB per file
_MAX_TOTAL_BYTES = 128 * 1024  # 128 KB total injection budget
_MAX_IMPORT_DEPTH = 5
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Match @path references: @file.md, @path/to/file, @~/home/file.md
# Stops at whitespace or common punctuation that isn't part of a path.
_IMPORT_RE = re.compile(r"@((?:~/)?[\w./\\-][\w./\\-]*\.[\w]+)")


# ── helpers ──────────────────────────────────────────────────────────────────


def _strip_html_comments(text: str) -> str:
    """Remove block-level HTML comments (<!-- … -->) from markdown."""
    return _HTML_COMMENT_RE.sub("", text)


def _safe_read(path: Path) -> Optional[str]:
    """Read a file if it exists, is a regular file, and under the size cap."""
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size == 0 or size > _MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text if text else None
    except (OSError, UnicodeDecodeError):
        return None


def _resolve_imports(
    text: str,
    base_dir: Path,
    seen: Set[str],
    depth: int = 0,
) -> str:
    """Expand @path references inline, up to _MAX_IMPORT_DEPTH levels deep."""
    if depth >= _MAX_IMPORT_DEPTH:
        return text

    def _replace(m: re.Match) -> str:
        raw_path = m.group(1).strip()
        if not raw_path:
            return m.group(0)
        # Expand ~ to home directory
        if raw_path.startswith("~"):
            target = Path(raw_path).expanduser().resolve()
        else:
            target = (base_dir / raw_path).resolve()
        key = str(target)
        if key in seen:
            return m.group(0)  # avoid cycles
        seen.add(key)
        content = _safe_read(target)
        if content is None:
            return m.group(0)  # leave reference as-is if file missing
        # Recursively expand imports in the imported file
        content = _resolve_imports(content, target.parent, seen, depth + 1)
        return content

    return _IMPORT_RE.sub(_replace, text)


def _load_file_with_imports(
    path: Path,
    seen: Set[str],
) -> Optional[str]:
    """Load a markdown file, resolve @imports, strip HTML comments."""
    key = str(path.resolve())
    if key in seen:
        return None
    seen.add(key)
    text = _safe_read(path)
    if text is None:
        return None
    text = _resolve_imports(text, path.parent, seen)
    text = _strip_html_comments(text)
    return text.strip() or None


def _collect_rules(rules_dir: Path, seen: Set[str]) -> List[str]:
    """Recursively collect all .md files from a rules/ directory."""
    parts: List[str] = []
    if not rules_dir.is_dir():
        return parts
    try:
        for md_file in sorted(rules_dir.rglob("*.md")):
            if not md_file.is_file():
                continue
            text = _load_file_with_imports(md_file, seen)
            if text:
                parts.append(text)
    except OSError:
        pass
    return parts


# ── public API ───────────────────────────────────────────────────────────────


def load_instruction_hierarchy(workspace_path: str | Path) -> tuple[str, set[str]]:
    """Load the full CLAUDE.md inheritance hierarchy for a workspace.

    Returns a tuple of (combined_text, loaded_paths) where loaded_paths
    is the set of resolved file paths that were loaded (for tracking
    on-demand subdirectory loading).
    """
    ws = Path(workspace_path).resolve()
    parts: List[str] = []
    seen: Set[str] = set()
    total_bytes = 0

    def _add(label: str, text: Optional[str]) -> None:
        nonlocal total_bytes
        if not text:
            return
        if total_bytes + len(text.encode("utf-8")) > _MAX_TOTAL_BYTES:
            log.warning("Instruction budget exceeded, skipping: %s", label)
            return
        total_bytes += len(text.encode("utf-8"))
        parts.append(f"# {label}\n\n{text}")

    # ── 1. User-level instructions ──────────────────────────────────────
    user_claude_dir = Path.home() / ".claude"

    user_claude_md = user_claude_dir / "CLAUDE.md"
    _add("User instructions (~/.claude/CLAUDE.md)",
         _load_file_with_imports(user_claude_md, seen))

    # User-level rules
    for rule_text in _collect_rules(user_claude_dir / "rules", seen):
        _add("User rule", rule_text)

    # ── 2. Ancestor directory walk (root → cwd) ────────────────────────
    # Walk from the workspace root's parents down to the workspace dir itself.
    # This mirrors Claude Code's "walk up from cwd" but we present them
    # in root→cwd order so more specific instructions come last (higher priority).
    ancestors: List[Path] = []
    current = ws
    while True:
        ancestors.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    ancestors.reverse()  # root first, workspace last

    for ancestor in ancestors:
        # Check <dir>/CLAUDE.md
        text = _load_file_with_imports(ancestor / "CLAUDE.md", seen)
        if text:
            rel = ancestor.relative_to(ws) if ancestor != ws and ancestor.is_relative_to(ws) else ancestor
            label = f"Instructions from {rel}/CLAUDE.md" if ancestor != ws else "Project CLAUDE.md"
            _add(label, text)

        # Check <dir>/.claude/CLAUDE.md (alternative location)
        text = _load_file_with_imports(ancestor / ".claude" / "CLAUDE.md", seen)
        if text:
            rel = ancestor.relative_to(ws) if ancestor != ws and ancestor.is_relative_to(ws) else ancestor
            label = f"Instructions from {rel}/.claude/CLAUDE.md" if ancestor != ws else "Project .claude/CLAUDE.md"
            _add(label, text)

    # ── 3. Project rules (.claude/rules/) ───────────────────────────────
    for rule_text in _collect_rules(ws / ".claude" / "rules", seen):
        _add("Project rule", rule_text)

    # ── 4. agent.md (harness-specific) ──────────────────────────────────
    _add("Agent instructions (agent.md)",
         _load_file_with_imports(ws / "agent.md", seen))

    if not parts:
        return "", seen
    return "\n\n".join(parts), seen


def load_subdirectory_instructions(
    workspace_path: str | Path,
    file_path: str | Path,
    already_loaded: Set[str],
) -> Optional[str]:
    """Load CLAUDE.md from a subdirectory when the agent reads a file there.

    This implements on-demand loading: when the agent reads a file in
    src/foo/bar/baz.py, we check for CLAUDE.md files in src/foo/bar/,
    src/foo/, and src/ (stopping at the workspace root).

    Returns new instruction text, or None if nothing new was found.
    """
    ws = Path(workspace_path).resolve()
    fp = Path(file_path).resolve()

    # Only handle files inside the workspace
    try:
        fp.relative_to(ws)
    except ValueError:
        return None

    parts: List[str] = []
    seen = set(already_loaded)

    # Walk from the file's parent up to (but not including) the workspace root
    current = fp.parent
    dirs_to_check: List[Path] = []
    while current != ws and str(current).startswith(str(ws)):
        dirs_to_check.append(current)
        current = current.parent
    dirs_to_check.reverse()  # process from shallowest to deepest

    for d in dirs_to_check:
        for name in ("CLAUDE.md", ".claude/CLAUDE.md"):
            candidate = d / name
            key = str(candidate.resolve())
            if key in seen:
                continue
            text = _load_file_with_imports(candidate, seen)
            if text:
                try:
                    rel = d.relative_to(ws)
                except ValueError:
                    rel = d
                parts.append(f"# Instructions from {rel}/{name}\n\n{text}")

        # Also check .claude/rules/ in subdirectories
        rules_dir = d / ".claude" / "rules"
        for rule_text in _collect_rules(rules_dir, seen):
            try:
                rel = d.relative_to(ws)
            except ValueError:
                rel = d
            parts.append(f"# Rule from {rel}/.claude/rules/\n\n{rule_text}")

    if not parts:
        return None
    return "\n\n".join(parts)
