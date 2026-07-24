"""CLAUDE.md loader — hierarchical project instructions, memory file attachment, path-scoped rules."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    User = "User"
    Project = "Project"
    Local = "Local"
    AutoMem = "AutoMem"

    def __str__(self) -> str:
        return self.value


@dataclass
class MemoryFileInfo:
    """Describes a loaded instruction/memory file."""

    path: str
    type: MemoryType
    content: str
    globs: Optional[List[str]] = None  # Conditional rule glob patterns
    parent: Optional[str] = None  # Path of file that @included this one
    raw_content: Optional[str] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INCLUDE_DEPTH = 5
MAX_MEMORY_CHAR_COUNT = 40000  # Warning threshold
MEMORY_INSTRUCTION_PROMPT = (
    "Codebase and user instructions are shown below. Be sure to adhere to "
    "these instructions. IMPORTANT: These instructions OVERRIDE any default "
    "behavior and you MUST follow them exactly as written."
)

# File extensions allowed for @include directives
TEXT_EXTENSIONS = {
    ".md", ".txt", ".text",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
    ".html", ".htm", ".css", ".scss", ".less",
    ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
    ".py", ".pyi", ".pyw",
    ".rb", ".erb", ".rake",
    ".go", ".rs", ".java", ".kt", ".kts", ".scala",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
    ".cs", ".swift",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".env", ".ini", ".cfg", ".conf",
    ".sql", ".graphql", ".gql",
    ".proto",
    ".vue", ".svelte",
    ".php", ".pl", ".pm", ".lua", ".r", ".R", ".dart",
    ".ex", ".exs", ".erl", ".hrl",
    ".hs", ".lhs", ".elm",
    ".ml", ".mli",
    ".rst", ".adoc", ".asciidoc", ".org", ".tex", ".latex",
    ".log", ".diff", ".patch",
}


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\s*\n([\s\S]*?)---\s*\n?", re.MULTILINE)


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict, remaining_content).
    """
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end():]

    frontmatter = {}
    for line in raw_yaml.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip("\"'")
            if key:
                frontmatter[key] = val

    return frontmatter, body


def parse_frontmatter_paths(raw_content: str) -> Tuple[str, Optional[List[str]]]:
    """Extract content and optional paths globs from frontmatter.

    Returns (content, paths_or_None).
    """
    fm, content = parse_frontmatter(raw_content)

    raw_paths = fm.get("paths")
    if not raw_paths:
        return content, None

    # Parse comma-separated or YAML list
    if isinstance(raw_paths, str):
        patterns = [p.strip() for p in raw_paths.split(",") if p.strip()]
    elif isinstance(raw_paths, list):
        patterns = [str(p).strip() for p in raw_paths if p]
    else:
        return content, None

    # Strip trailing /** — ignore library handles dirs naturally
    cleaned = []
    for p in patterns:
        p = p.rstrip("/")
        if p.endswith("/**"):
            p = p[:-3]
        if p and p != "**":
            cleaned.append(p)

    if not cleaned:
        return content, None

    return content, cleaned


# ---------------------------------------------------------------------------
# HTML comment stripping
# ---------------------------------------------------------------------------


def strip_html_comments(content: str) -> Tuple[str, bool]:
    """Strip block-level HTML comments from markdown content.

    Only strips comments that occupy their own block (CommonMark html blocks),
    not inline comments within paragraphs or code blocks.

    Returns (cleaned_content, was_stripped).
    """
    if "<!--" not in content:
        return content, False

    # Simple approach: strip comment-like blocks that start at beginning of line
    lines = content.split("\n")
    result = []
    in_comment = False
    stripped = False

    for line in lines:
        if not in_comment and "<!--" in line:
            idx = line.index("<!--")
            # Check if it looks like a block comment (at start or after whitespace)
            prefix = line[:idx]
            if not prefix.strip():
                # Block-level comment — strip it
                if "-->" in line[idx:]:
                    # Single-line comment
                    after = line[idx + 4:]
                    end_idx = after.index("-->")
                    residue = after[end_idx + 3:]
                    if residue.strip():
                        result.append(prefix + residue)
                    stripped = True
                else:
                    in_comment = True
                    # Strip from comment start, keep anything before
                    result.append(prefix)
                continue

        if in_comment:
            if "-->" in line:
                idx = line.index("-->")
                after = line[idx + 3:]
                if after.strip():
                    result.append(after)
                in_comment = False
                stripped = True
            continue

        result.append(line)

    if in_comment:
        # Unclosed comment — keep everything
        return content, False

    return "\n".join(result), stripped


# ---------------------------------------------------------------------------
# @include directive extraction
# ---------------------------------------------------------------------------

INCLUDE_RE = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")


def extract_include_paths(content: str, base_path: str) -> List[str]:
    """Extract @include paths from markdown content.

    Handles @path, @./relative, @~/home, @/absolute.
    Skips paths inside code blocks (```...```) and inline code (`...`).
    """
    paths: Set[str] = set()

    # Strip code blocks before scanning
    stripped = re.sub(r"```[\s\S]*?```", "", content)
    stripped = re.sub(r"`[^`]*`", "", stripped)

    for match in INCLUDE_RE.finditer(stripped):
        raw_path = match.group(1)
        if not raw_path:
            continue

        # Strip fragment identifiers (#heading)
        if "#" in raw_path:
            raw_path = raw_path[: raw_path.index("#")]

        if not raw_path:
            continue

        # Unescape spaces
        raw_path = raw_path.replace("\\ ", " ")

        # Validate path format
        if not (
            raw_path.startswith(("./", "~/")) or
            raw_path.startswith("/") or
            (raw_path[0].isalpha() or raw_path[0] in "._-")
        ):
            continue

        # Resolve the path
        if raw_path.startswith("~/"):
            resolved = os.path.expanduser(raw_path)
        elif raw_path.startswith("/"):
            resolved = raw_path
        else:
            # Relative to the directory of the including file
            base_dir = os.path.dirname(base_path) if os.path.isabs(base_path) else Path.cwd()
            if raw_path.startswith("./"):
                raw_path = raw_path[2:]
            resolved = os.path.normpath(os.path.join(base_dir, raw_path))

        # Only allow text file extensions
        ext = os.path.splitext(resolved)[1].lower()
        if ext and ext not in TEXT_EXTENSIONS:
            continue

        paths.add(resolved)

    return sorted(paths)


# ---------------------------------------------------------------------------
# Memory file processing
# ---------------------------------------------------------------------------

def _read_file_safe(filepath: str) -> Optional[str]:
    """Read a file, returning None if it doesn't exist or can't be read."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return None
    except Exception as e:
        logger.debug("Error reading %s: %s", filepath, e)
        return None


def parse_memory_file_content(
    raw_content: str,
    filepath: str,
    mem_type: MemoryType,
    include_base_path: Optional[str] = None,
) -> Tuple[Optional[MemoryFileInfo], List[str]]:
    """Parse raw memory file content into a MemoryFileInfo.

    Returns (info_or_None, include_paths).
    """
    # Skip non-text files
    ext = os.path.splitext(filepath)[1].lower()
    if ext and ext not in TEXT_EXTENSIONS:
        logger.debug("Skipping non-text file in @include: %s", filepath)
        return None, []

    # Parse frontmatter and paths
    content_without_fm, paths = parse_frontmatter_paths(raw_content)

    # Strip HTML comments
    stripped_content, _ = strip_html_comments(content_without_fm)

    info = MemoryFileInfo(
        path=filepath,
        type=mem_type,
        content=stripped_content,
        globs=paths,
    )

    # Extract @include paths
    include_paths = []
    if include_base_path is not None:
        include_paths = extract_include_paths(stripped_content, include_base_path)

    return info, include_paths


async def process_memory_file(
    filepath: str,
    mem_type: MemoryType,
    processed_paths: Set[str],
    include_external: bool = False,
    depth: int = 0,
    parent: Optional[str] = None,
) -> List[MemoryFileInfo]:
    """Recursively process a memory file and its @include references.

    Returns list of MemoryFileInfo (includes first, then main file).
    """
    # Normalize for dedup
    normalized = os.path.normpath(os.path.realpath(filepath)) if os.path.exists(filepath) else filepath
    if normalized in processed_paths or depth >= MAX_INCLUDE_DEPTH:
        return []

    processed_paths.add(normalized)

    raw_content = _read_file_safe(filepath)
    if not raw_content or not raw_content.strip():
        return []

    info, include_paths = parse_memory_file_content(
        raw_content, filepath, mem_type,
        include_base_path=filepath if depth == 0 else None,
    )
    if info is None:
        return []

    if parent:
        info.parent = parent

    result: List[MemoryFileInfo] = []
    result.append(info)

    for include_path in include_paths:
        # Check if external (outside CWD)
        cwd = str(Path.cwd())
        is_external = not os.path.abspath(include_path).startswith(os.path.abspath(cwd).rstrip("/\\") + os.sep)
        if is_external and not include_external:
            continue

        included = await process_memory_file(
            include_path, mem_type, processed_paths,
            include_external, depth + 1, filepath,
        )
        result.extend(included)

    return result


async def process_md_rules(
    rules_dir: str,
    mem_type: MemoryType,
    processed_paths: Set[str],
    include_external: bool = False,
    conditional_rule: bool = False,
) -> List[MemoryFileInfo]:
    """Process all .md files in a .claude/rules/ directory (recursive).

    When conditional_rule=True, only includes files WITH frontmatter paths.
    When conditional_rule=False, only includes files WITHOUT frontmatter paths.
    """
    rules_path = Path(rules_dir)
    if not rules_path.exists() or not rules_path.is_dir():
        return []

    result: List[MemoryFileInfo] = []

    for entry in sorted(rules_path.rglob("*.md")):
        entry_str = str(entry)

        try:
            raw = entry.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        _, paths = parse_frontmatter_paths(raw)

        has_globs = paths is not None and len(paths) > 0

        # Filter: conditional_rule=True → only files WITH paths
        #          conditional_rule=False → only files WITHOUT paths
        if conditional_rule != has_globs:
            continue

        files = await process_memory_file(
            entry_str, mem_type, processed_paths, include_external,
        )
        result.extend(files)

    return result


# ---------------------------------------------------------------------------
# Conditional rule matching
# ---------------------------------------------------------------------------

import fnmatch


def _path_matches_globs(target_path: str, globs: List[str], base_dir: str) -> bool:
    """Check if a target path matches any of the given glob patterns.

    Globs are resolved relative to base_dir.
    """
    try:
        rel = os.path.relpath(target_path, base_dir)
    except ValueError:
        return False

    # Normalize to forward slashes for matching
    rel = rel.replace("\\", "/")

    for pattern in globs:
        pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(rel, pattern):
            return True
        # Also check if rel starts with the pattern dir
        if pattern.endswith("/*") or pattern.endswith("/**"):
            prefix = pattern[:-2]
            if rel.startswith(prefix.lstrip("/") + "/"):
                return True

    return False


async def process_conditioned_md_rules(
    target_path: str,
    rules_dir: str,
    mem_type: MemoryType,
    processed_paths: Set[str],
    include_external: bool = False,
) -> List[MemoryFileInfo]:
    """Process conditional rules matching target_path.

    Only includes .md files whose paths: globs match the target path.
    """
    all_rules = await process_md_rules(
        rules_dir, mem_type, processed_paths,
        include_external, conditional_rule=True,
    )

    # Resolve base directory
    rules_path = Path(rules_dir)
    # For Project rules: base is parent of .claude/
    # For User rules: base is CWD
    if mem_type == MemoryType.Project:
        base_dir = str(rules_path.parent.parent)  # .claude/rules/ → .claude/ → parent
    else:
        base_dir = str(Path.cwd())

    matched = []
    for rule in all_rules:
        if rule.globs and _path_matches_globs(target_path, rule.globs, base_dir):
            matched.append(rule)

    return matched


# ---------------------------------------------------------------------------
# Main loading: getMemoryFiles
# ---------------------------------------------------------------------------

_GET_MEMORY_FILES_CACHE: Optional[List[MemoryFileInfo]] = None
_NEXT_EAGER_LOAD_REASON: str = "session_start"
_SHOULD_FIRE_HOOK: bool = True


def _get_user_claude_dir() -> Path:
    """Get ~/.claude directory."""
    return Path.home() / ".claude"


def _get_user_memory_path() -> Path:
    """Get ~/.claude/CLAUDE.md."""
    return _get_user_claude_dir() / "CLAUDE.md"


def _get_user_rules_dir() -> Path:
    """Get ~/.claude/rules/."""
    return _get_user_claude_dir() / "rules"


# ---------------------------------------------------------------------------
# Nested / lazy loading
# ---------------------------------------------------------------------------


async def get_managed_and_user_conditional_rules(
    target_path: str,
    processed_paths: Set[str],
) -> List[MemoryFileInfo]:
    """Get User conditional rules matching a target path.

    (No Managed tier in harness — just User-level conditional rules.)
    """
    result: List[MemoryFileInfo] = []

    user_rules = str(_get_user_rules_dir())
    result.extend(
        await process_conditioned_md_rules(
            target_path, user_rules, MemoryType.User, processed_paths, True,
        )
    )

    return result


async def get_memory_files_for_nested_directory(
    dir_path: str,
    target_path: str,
    processed_paths: Set[str],
) -> List[MemoryFileInfo]:
    """Get memory files for a nested directory (between CWD and target).

    Loads CLAUDE.md, unconditional rules, and conditional rules for that directory.
    """
    result: List[MemoryFileInfo] = []
    d = Path(dir_path)

    # Project memory files
    for name in ("CLAUDE.md", ".claude/CLAUDE.md"):
        p = d / name
        if p.exists():
            result.extend(
                await process_memory_file(str(p), MemoryType.Project, processed_paths, False)
            )

    # Local
    local_path = d / "CLAUDE.local.md"
    if local_path.exists():
        result.extend(
            await process_memory_file(str(local_path), MemoryType.Local, processed_paths, False)
        )

    # Unconditional rules
    rules_dir = d / ".claude" / "rules"
    if rules_dir.exists():
        unconditional = await process_md_rules(
            str(rules_dir), MemoryType.Project, processed_paths, False, False
        )
        result.extend(unconditional)

        # Conditional rules
        conditional = await process_conditioned_md_rules(
            target_path, str(rules_dir), MemoryType.Project, processed_paths, False
        )
        result.extend(conditional)

    return result


# ---------------------------------------------------------------------------
# Formatting for context injection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------



