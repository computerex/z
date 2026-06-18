"""
Memory system (memdir) — file-based persistent memory for Claude Code.

Mirrors Claude Code's memdir subsystem:
- File-based storage at ~/.claude/projects/<slug>/memory/
- MEMORY.md index + topic files pattern
- 4-type closed taxonomy (user, feedback, project, reference)
- Cross-encoder based relevance ranking for memory recall

Memory directory structure:
  ~/.claude/projects/<project-slug>/memory/
  ├── MEMORY.md              # Always-loaded index file
  ├── user_role.md           # Topic files with frontmatter
  ├── feedback_testing.md
  └── ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

MEMORY_TYPES = ["user", "feedback", "project", "reference"]

AUTO_MEM_DISPLAY_NAME = "auto memory"

# Ensure directory guidance
DIR_EXISTS_GUIDANCE = (
    "This directory already exists — write to it directly with the Write tool "
    "(do not run mkdir or check for its existence)."
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class MemoryHeader:
    """Header info for a memory file (from frontmatter)."""

    filename: str
    file_path: str
    mtime_ms: float
    description: Optional[str]
    mem_type: Optional[str]  # "user", "feedback", "project", "reference"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_claude_dir() -> Path:
    """Get ~/.claude directory."""
    return Path.home() / ".claude"


def get_projects_dir() -> Path:
    """Get ~/.claude/projects/ directory."""
    return get_claude_dir() / "projects"


def _compute_project_slug(cwd: Optional[str] = None) -> str:
    """Compute a stable slug for the current project directory.

    Uses dirname-based hashing to produce a unique slug.
    """
    project_path = Path(cwd or os.getcwd()).resolve()
    # Use the directory name as the slug, with a hash suffix for uniqueness
    name = project_path.name
    h = hashlib.md5(str(project_path).encode()).hexdigest()
    return f"{name}_{h[:12]}"


def get_memory_dir(cwd: Optional[str] = None) -> Path:
    """Get the memory directory path for the current project."""
    return get_projects_dir() / _compute_project_slug(cwd) / "memory"


def get_memory_entrypoint(cwd: Optional[str] = None) -> Path:
    """Get the MEMORY.md path."""
    return get_memory_dir(cwd) / ENTRYPOINT_NAME


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


def ensure_memory_dir_exists(memory_dir: Optional[str] = None) -> Path:
    """Ensure the memory directory exists. Idempotent."""
    if memory_dir:
        d = Path(memory_dir)
    else:
        d = get_memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Entrypoint truncation
# ---------------------------------------------------------------------------


@dataclass
class EntrypointTruncation:
    """Result of truncating MEMORY.md content."""

    content: str
    line_count: int
    byte_count: int
    was_line_truncated: bool
    was_byte_truncated: bool


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    """Truncate MEMORY.md content to line AND byte caps."""
    trimmed = raw.strip()
    content_lines = trimmed.split("n")
    line_count = len(content_lines)
    byte_count = len(trimmed.encode("utf-8"))

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(
            content=trimmed,
            line_count=line_count,
            byte_count=byte_count,
            was_line_truncated=False,
            was_byte_truncated=False,
        )

    # Line truncate first
    truncated = "n".join(content_lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed

    # Byte truncate if still too large
    truncated_bytes = truncated.encode("utf-8")
    if len(truncated_bytes) > MAX_ENTRYPOINT_BYTES:
        # Cut at last newline before the byte cap
        cut = truncated.rfind("n", 0, MAX_ENTRYPOINT_BYTES)
        if cut > 0:
            truncated = truncated[:cut]
        else:
            truncated = truncated[:MAX_ENTRYPOINT_BYTES]

    # Format size info
    size_str = f"{byte_count:,} bytes (limit: {MAX_ENTRYPOINT_BYTES:,} bytes)" if was_byte_truncated else ""
    line_str = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})" if was_line_truncated else ""
    if line_str and size_str:
        reason = f"{line_str} and {size_str}"
    else:
        reason = line_str or size_str

    return EntrypointTruncation(
        content=(
            truncated.strip()
            + f"nn> WARNING: {ENTRYPOINT_NAME} is {reason}. "
            "Only part of it was loaded. Keep index entries to one line "
            "under ~200 chars; move detail into topic files."
        ),
        line_count=line_count,
        byte_count=byte_count,
        was_line_truncated=was_line_truncated,
        was_byte_truncated=was_byte_truncated,
    )


# ---------------------------------------------------------------------------
# Memory prompt sections
# ---------------------------------------------------------------------------

MEMORY_TYPES_SECTION = """## Types of memory

There are several discrete types of memory that you can store in your memory system:

<types>
<type>
    <name>user</name>
    <description>Contains information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective.</description>
    <when_to_save>When you learn any details about the user's role, preferences, responsibilities, or knowledge</when_to_save>
    <how_to_use>When your work should be informed by the user's profile or perspective.</how_to_use>
</type>
<type>
    <name>feedback</name>
    <description>Guidance the user has given you about how to approach work — both what to avoid and what to keep doing. These are a very important type of memory to read and write as they allow you to remain coherent and responsive to the way you should approach work in the project.</description>
    <when_to_save>Any time the user corrects your approach OR confirms a non-obvious approach worked. Include *why* so you can judge edge cases later.</when_to_save>
    <how_to_use>Let these memories guide your behavior so that the user does not need to offer the same guidance twice.</how_to_use>
</type>
<type>
    <name>project</name>
    <description>Information that you learn about ongoing work, goals, initiatives, bugs, or incidents within the project that is not otherwise derivable from the code or git history.</description>
    <when_to_save>When you learn who is doing what, why, or by when. Always convert relative dates to absolute dates.</when_to_save>
    <how_to_use>Use these memories to more fully understand the context behind the user's request and make better informed suggestions.</how_to_use>
</type>
<type>
    <name>reference</name>
    <description>Stores pointers to where information can be found in external systems. These memories allow you to remember where to look to find up-to-date information outside of the project directory.</description>
    <when_to_save>When you learn about resources in external systems and their purpose.</when_to_save>
    <how_to_use>When the user references an external system or information that may be in an external system.</how_to_use>
</type>
</types>"""

WHAT_NOT_TO_SAVE_SECTION = """## What NOT to save in memory

- Code patterns, conventions, architecture, file paths, or project structure — these can be derived by reading the current project state.
- Git history, recent changes, or who-changed-what — `git log` / `git blame` are authoritative.
- Debugging solutions or fix recipes — the fix is in the code; the commit message has the context.
- Anything already documented in CLAUDE.md files.
- Ephemeral task details: in-progress work, temporary state, current conversation context.

These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it — that is the part worth keeping."""

WHEN_TO_ACCESS_SECTION = """## When to access memories

- When memories seem relevant, or the user references prior-conversation work.
- You MUST access memory when the user explicitly asks you to check, recall, or remember.
- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.
- Memory records can become stale over time. Use memory as context for what was true at a given point in time. Before answering the user or building assumptions based solely on information in memory records, verify that the memory is still correct and up-to-date by reading the current state of the files or resources."""

TRUSTING_RECALL_SECTION = """## Before recommending from memory

A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:

- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation (not just asking about history), verify first.

"The memory says X exists" is not the same as "X exists now."

A memory that summarizes repo state (activity logs, architecture snapshots) is frozen in time. If the user asks about *recent* or *current* state, prefer `git log` or reading the code over recalling the snapshot."""

MEMORY_FRONTMATTER_EXAMPLE = r"""```markdown
---
name: {{memory name}}
description: {{one-line description — used to decide relevance in future conversations, so be specific}}
type: {{user, feedback, project, reference}}
---

{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}
```"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_memory_prompt(
    memory_dir: str,
    extra_guidelines: Optional[List[str]] = None,
) -> str:
    """Build the typed-memory behavioral instructions with MEMORY.md content.

    This is injected into the system prompt so the model understands
    how to use the memory system.
    """
    md_path = os.path.join(memory_dir, ENTRYPOINT_NAME)

    # Read existing entrypoint
    entrypoint_content = ""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            entrypoint_content = f.read()
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("Error reading %s: %s", md_path, e)

    lines = [
        f"# {AUTO_MEM_DISPLAY_NAME}",
        "",
        f"You have a persistent, file-based memory system at `{memory_dir}`. {DIR_EXISTS_GUIDANCE}",
        "",
        "You should build up this memory system over time so that future conversations "
        "can have a complete picture of who the user is, how they'd like to collaborate "
        "with you, what behaviors to avoid or repeat, and the context behind the work "
        "the user gives you.",
        "",
        "If the user explicitly asks you to remember something, save it immediately as "
        "whichever type fits best. If they ask you to forget something, find and remove "
        "the relevant entry.",
        "",
        MEMORY_TYPES_SECTION,
        "",
        WHAT_NOT_TO_SAVE_SECTION,
        "",
        "## How to save memories",
        "",
        "Saving a memory is a two-step process:",
        "",
        "**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) "
        "using this frontmatter format:",
        "",
        MEMORY_FRONTMATTER_EXAMPLE,
        "",
        f"**Step 2** — add a pointer to that file in `{ENTRYPOINT_NAME}`. `{ENTRYPOINT_NAME}` is "
        "an index, not a memory — each entry should be one line, under ~150 characters: "
        "`- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory "
        "content directly into `{ENTRYPOINT_NAME}`.",
        "",
        f"- `{ENTRYPOINT_NAME}` is always loaded into your conversation context — lines "
        f"after {MAX_ENTRYPOINT_LINES} will be truncated, so keep the index concise",
        "- Keep the name, description, and type fields in memory files up-to-date with the content",
        "- Organize memory semantically by topic, not chronologically",
        "- Update or remove memories that turn out to be wrong or outdated",
        "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        "",
        WHEN_TO_ACCESS_SECTION,
        "",
        TRUSTING_RECALL_SECTION,
        "",
        "## Memory and other forms of persistence",
        "",
        "Memory is one of several persistence mechanisms available to you as you assist the "
        "user in a given conversation. The distinction is often that memory can be recalled "
        "in future conversations and should not be used for persisting information that is "
        "only useful within the scope of the current conversation.",
        "",
        "- When to use or update a plan instead of memory: If you are about to start a "
        "non-trivial implementation task and would like to reach alignment with the user on "
        "your approach you should use a Plan rather than saving this information to memory.",
        "- When to use or update tasks instead of memory: When you need to break your work "
        "in current conversation into discrete steps or keep track of your progress use "
        "tasks instead of saving to memory.",
        "",
    ]

    # Build searching past context section
    lines.extend([
        "## Searching past context",
        "",
        "When looking for past context:",
        "1. Search topic files in your memory directory:",
        "```",
        f'grep -rn "<search term>" {memory_dir} --include="*.md"',
        "```",
        "Use narrow search terms (error messages, file paths, function names) "
        "rather than broad keywords.",
        "",
    ])

    # Append entrypoint content
    if entrypoint_content.strip():
        t = truncate_entrypoint_content(entrypoint_content)
        lines.extend([
            "",
            f"## {ENTRYPOINT_NAME}",
            "",
            t.content,
        ])
    else:
        lines.extend([
            "",
            f"## {ENTRYPOINT_NAME}",
            "",
            f"Your {ENTRYPOINT_NAME} is currently empty. When you save new memories, they will appear here.",
        ])

    # Extra guidelines
    if extra_guidelines:
        lines.extend(["", *extra_guidelines, ""])

    return "n".join(lines)


# ---------------------------------------------------------------------------
# Memory directory scanning
# ---------------------------------------------------------------------------


def scan_memory_files(
    memory_dir: str,
) -> List[MemoryHeader]:
    """Scan a memory directory for .md files and read their frontmatter.

    Returns list of MemoryHeader sorted newest-first.
    """
    mem_path = Path(memory_dir)
    if not mem_path.exists():
        return []

    headers: List[MemoryHeader] = []

    for fpath in mem_path.rglob("*.md"):
        if fpath.name == ENTRYPOINT_NAME:
            continue

        try:
            stat = fpath.stat()
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Parse frontmatter
        fm = _parse_simple_frontmatter(content)

        description = fm.get("description") if isinstance(fm.get("description"), str) else None
        mem_type = fm.get("type") if isinstance(fm.get("type"), str) and fm["type"] in MEMORY_TYPES else None

        headers.append(MemoryHeader(
            filename=fpath.name,
            file_path=str(fpath),
            mtime_ms=stat.st_mtime * 1000,
            description=description,
            mem_type=mem_type,
        ))

    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers


def _parse_simple_frontmatter(content: str) -> dict:
    """Quick frontmatter parser for memory files (no YAML dependency needed)."""
    match = re.match(r"^---s*n([sS]*?)---s*n?", content)
    if not match:
        return {}

    result = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip("\"'")
    return result


def format_memory_manifest(headers: List[MemoryHeader]) -> str:
    """Format memory headers as a text manifest for the ranking query."""
    lines = []
    for m in headers:
        tag = f"[{m.mem_type}] " if m.mem_type else ""
        ts = _format_timestamp(m.mtime_ms)
        if m.description:
            lines.append(f"- {tag}{m.filename} ({ts}): {m.description}")
        else:
            lines.append(f"- {tag}{m.filename} ({ts})")
    return "n".join(lines)


def _format_timestamp(mtime_ms: float) -> str:
    """Format a millisecond timestamp as ISO date."""
    import datetime
    dt = datetime.datetime.fromtimestamp(mtime_ms / 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Memory recall via cross-encoder
# ---------------------------------------------------------------------------


def find_relevant_memories(
    query: str,
    memory_dir: str,
    top_k: int = 5,
) -> List[str]:
    """Find memory files relevant to a query using cross-encoder ranking.

    Returns absolute file paths of the most relevant memories (up to top_k).
    Falls back to keyword matching if the cross-encoder is unavailable.
    """
    memories = scan_memory_files(memory_dir)
    if not memories:
        return []

    from harness.cross_encoder import MemoryCandidate, rank_memories

    candidates = [
        MemoryCandidate(
            filepath=m.file_path,
            description=m.description or "",
            content_preview=_read_content_preview(m.file_path),
        )
        for m in memories
    ]

    ranked = rank_memories(query, candidates, top_k)
    return [c.filepath for c in ranked if c.score > 0]


def _read_content_preview(filepath: str, max_chars: int = 300) -> str:
    """Read first part of a memory file (skipping frontmatter)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Strip frontmatter
        match = re.match(r"^---s*n[sS]*?---s*n?", content)
        if match:
            content = content[match.end():]

        content = content.strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        return content
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Convenience: load memory prompt directly
# ---------------------------------------------------------------------------


def load_memory_prompt() -> Optional[str]:
    """Load the unified memory prompt for inclusion in the system prompt.

    Returns None if memory is disabled or unavailable.
    """
    mem_dir = get_memory_dir()
    ensure_memory_dir_exists(str(mem_dir))
    return build_memory_prompt(str(mem_dir))
