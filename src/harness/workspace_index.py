"""Workspace file index — built at startup for project-aware context.

Provides the agent with a map of the codebase so it knows what files
exist without needing to call list_files repeatedly.  Uses `git ls-files`
when available (fast, respects .gitignore), falls back to a filtered
filesystem walk.
"""

import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .logger import get_logger

log = get_logger("index")

# Directories to skip during manual walk (when git is unavailable)
SKIP_DIRS = frozenset({
    '.git', '.hg', '.svn', 'node_modules', '__pycache__', '.venv', 'venv',
    'env', '.env', 'dist', 'build', 'target', 'vendor', 'obj', 'bin',
    '.tox', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.next',
    '.nuxt', '.output', 'coverage', '.coverage', '.idea', '.vscode',
    '.harness_output', '.sessions',
})

# Binary / non-text extensions to exclude from line counting
BINARY_EXTENSIONS = frozenset({
    '.exe', '.dll', '.so', '.dylib', '.o', '.a', '.lib',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.wav', '.ogg', '.webm', '.avi', '.mov',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.bin', '.dat', '.db', '.sqlite', '.sqlite3',
    '.woff', '.woff2', '.ttf', '.otf', '.eot',
    '.pyc', '.pyo', '.class', '.jar',
    '.model', '.onnx', '.pt', '.pth', '.h5', '.tflite',
})

# Language detection by extension
LANG_MAP = {
    '.py': 'Python', '.pyw': 'Python',
    '.js': 'JavaScript', '.mjs': 'JavaScript', '.cjs': 'JavaScript',
    '.ts': 'TypeScript', '.tsx': 'TypeScript', '.jsx': 'JavaScript',
    '.go': 'Go',
    '.rs': 'Rust',
    '.java': 'Java', '.kt': 'Kotlin', '.scala': 'Scala',
    '.c': 'C', '.h': 'C', '.cpp': 'C++', '.cc': 'C++', '.hpp': 'C++',
    '.cs': 'C#',
    '.rb': 'Ruby',
    '.php': 'PHP',
    '.swift': 'Swift',
    '.r': 'R', '.R': 'R',
    '.lua': 'Lua',
    '.sh': 'Shell', '.bash': 'Shell', '.zsh': 'Shell', '.fish': 'Shell',
    '.ps1': 'PowerShell', '.psm1': 'PowerShell',
    '.sql': 'SQL',
    '.html': 'HTML', '.htm': 'HTML',
    '.css': 'CSS', '.scss': 'SCSS', '.less': 'Less',
    '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML',
    '.xml': 'XML',
    '.md': 'Markdown', '.rst': 'reStructuredText',
    '.dockerfile': 'Docker',
    '.proto': 'Protobuf',
    '.graphql': 'GraphQL', '.gql': 'GraphQL',
    '.tf': 'Terraform', '.hcl': 'HCL',
    '.vue': 'Vue', '.svelte': 'Svelte',
}


class FileInfo:
    """Lightweight file metadata."""
    __slots__ = ('rel_path', 'size', 'lines', 'extension', 'language', 'is_binary')

    def __init__(self, rel_path: str, size: int, lines: int,
                 extension: str, language: str, is_binary: bool):
        self.rel_path = rel_path
        self.size = size
        self.lines = lines
        self.extension = extension
        self.language = language
        self.is_binary = is_binary


class WorkspaceIndex:
    """Index of files in the workspace, built at startup."""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.files: List[FileInfo] = []
        self._by_dir: Dict[str, List[FileInfo]] = defaultdict(list)
        self._build_time: float = 0.0
        self._is_git: bool = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self) -> "WorkspaceIndex":
        """Build the index (synchronous — called once at startup)."""
        t0 = time.time()
        ws = Path(self.workspace_path)

        # Try git ls-files first (fastest, respects .gitignore)
        file_list = self._git_ls_files(ws)
        if file_list is not None:
            self._is_git = True
            log.info("Index: using git ls-files (%d files)", len(file_list))
        else:
            file_list = self._walk_filesystem(ws)
            log.info("Index: using filesystem walk (%d files)", len(file_list))

        # Build file info for each path
        self.files = []
        self._by_dir = defaultdict(list)

        for rel_str in file_list:
            abs_path = ws / rel_str
            ext = os.path.splitext(rel_str)[1].lower()
            is_binary = ext in BINARY_EXTENSIONS
            lang = LANG_MAP.get(ext, '')

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue  # file vanished

            # Count lines for text files (cheap — just count newlines)
            lines = 0
            if not is_binary and size < 2_000_000:  # skip files > 2 MB
                try:
                    lines = abs_path.read_bytes().count(b'\n')
                except OSError:
                    pass

            info = FileInfo(
                rel_path=rel_str.replace('\\', '/'),  # normalise to forward slashes
                size=size,
                lines=lines,
                extension=ext,
                language=lang,
                is_binary=is_binary,
            )
            self.files.append(info)

            # Group by parent directory
            parent = os.path.dirname(info.rel_path) or '.'
            self._by_dir[parent].append(info)

        self._build_time = time.time() - t0
        log.info("Index built: %d files in %.2fs", len(self.files), self._build_time)
        return self

    def _git_ls_files(self, ws: Path) -> Optional[List[str]]:
        """Use `git ls-files` to get tracked files. Returns None if not a git repo."""
        try:
            result = subprocess.run(
                ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                capture_output=True, text=True, cwd=str(ws), timeout=10,
            )
            if result.returncode != 0:
                return None
            files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
            # Filter out binary files that git tracks (e.g. images checked in)
            return [f for f in files if os.path.isfile(ws / f)]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _walk_filesystem(self, ws: Path) -> List[str]:
        """Manual filesystem walk with skip patterns."""
        result = []
        for root, dirs, files in os.walk(ws, topdown=True):
            # Filter dirs in-place to skip junk
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

            rel_root = os.path.relpath(root, ws)
            for fname in files:
                if fname.startswith('.'):
                    continue
                rel_path = os.path.join(rel_root, fname) if rel_root != '.' else fname
                result.append(rel_path)

            if len(result) > 10_000:
                log.warning("Index walk truncated at 10,000 files")
                break

        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def search(self, pattern: str) -> List[FileInfo]:
        """Search file paths by substring (case-insensitive)."""
        pattern_lower = pattern.lower()
        return [f for f in self.files if pattern_lower in f.rel_path.lower()]

    def get_dir(self, directory: str) -> List[FileInfo]:
        """Get files in a specific directory."""
        norm = directory.replace('\\', '/').strip('/')
        if norm == '' or norm == '.':
            # Root-level files only
            return self._by_dir.get('.', [])
        return self._by_dir.get(norm, [])

    def get_languages(self) -> Dict[str, int]:
        """Get a count of files per language."""
        counts: Dict[str, int] = defaultdict(int)
        for f in self.files:
            if f.language:
                counts[f.language] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Summary for system prompt
    # ------------------------------------------------------------------

    def summary(self, max_lines: int = 80) -> str:
        """Generate a compact project map for the system prompt.

        Produces a tree grouped by directory with file counts and
        language breakdown.  Stays within *max_lines* to keep the
        system prompt budget reasonable.
        """
        if not self.files:
            return "(workspace index: no files found)"

        parts: List[str] = []
        langs = self.get_languages()
        total_lines = sum(f.lines for f in self.files if not f.is_binary)

        # Header
        lang_summary = ", ".join(f"{lang} ({n})" for lang, n in list(langs.items())[:6])
        if len(langs) > 6:
            lang_summary += f", +{len(langs) - 6} more"
        parts.append(
            f"PROJECT MAP — {len(self.files)} files, "
            f"~{total_lines:,} lines | {lang_summary}"
        )
        if self._is_git:
            parts.append("(git repo — .gitignore respected)")
        parts.append("")

        # Build directory tree
        lines_used = len(parts)
        remaining = max_lines - lines_used - 2  # reserve footer

        # Sort directories: root first, then alphabetical
        dir_order = sorted(self._by_dir.keys(), key=lambda d: (d != '.', d.lower()))

        for dir_name in dir_order:
            if remaining <= 0:
                parts.append(f"  ... ({len(dir_order) - dir_order.index(dir_name)} more directories)")
                break

            dir_files = self._by_dir[dir_name]
            dir_label = dir_name + '/' if dir_name != '.' else '(root)'

            # For small directories, list individual files
            if len(dir_files) <= 5:
                parts.append(f"{dir_label}")
                for f in sorted(dir_files, key=lambda x: x.rel_path):
                    fname = os.path.basename(f.rel_path)
                    detail = f"{f.lines}L" if f.lines else f"{f.size}B"
                    lang_tag = f" [{f.language}]" if f.language else ""
                    parts.append(f"  {fname} ({detail}{lang_tag})")
                remaining -= 1 + len(dir_files)
            else:
                # Summarise large directories
                text_files = [f for f in dir_files if not f.is_binary]
                dir_lines = sum(f.lines for f in text_files)
                dir_langs = set(f.language for f in dir_files if f.language)
                lang_str = ", ".join(sorted(dir_langs)[:3])
                parts.append(
                    f"{dir_label} — {len(dir_files)} files, "
                    f"~{dir_lines:,}L"
                    + (f" [{lang_str}]" if lang_str else "")
                )
                # Show top 3 largest files
                top = sorted(text_files, key=lambda f: f.lines, reverse=True)[:3]
                for f in top:
                    fname = os.path.basename(f.rel_path)
                    parts.append(f"  {fname} ({f.lines}L)")
                remaining -= 1 + min(3, len(top))

        # Footer
        parts.append("")
        parts.append(f"Index built in {self._build_time:.2f}s")

        return "\n".join(parts)

    def compact_tree(self) -> str:
        """Return a minimal file list (paths only) for tight budgets."""
        paths = sorted(f.rel_path for f in self.files if not f.is_binary)
        if len(paths) > 200:
            return "\n".join(paths[:200]) + f"\n... (+{len(paths) - 200} more files)"
        return "\n".join(paths)

