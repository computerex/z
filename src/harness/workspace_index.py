"""Workspace file index — built at startup for project-aware context.

Provides the agent with a map of the codebase so it knows what files
exist without needing to call list_files repeatedly.  Uses `git ls-files`
when available (fast, respects .gitignore), falls back to a smart
shallow scan that respects .gitignore via pathspec and limits depth.
"""

import os
import subprocess
import time
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .logger import get_logger

try:
    import pathspec
    HAS_PATHSPEC = True
except ImportError:
    pathspec = None
    HAS_PATHSPEC = False

log = get_logger("index")

# Directories to always skip (even if not in .gitignore)
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

# Max files to detail-scan per directory at depth > 1
_DEEP_DIR_FILE_CAP = 10

# Max total files before we skip line counting entirely
_LINE_COUNT_FILE_CAP = 500

# Max depth for the shallow scan when NOT in a git repo
_MAX_SCAN_DEPTH = 4

# Files that indicate a directory is a project root
_PROJECT_MARKERS = frozenset({
    '.git', 'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod',
    'pom.xml', 'build.gradle', 'Makefile', 'CMakeLists.txt', 'setup.py',
    'requirements.txt', 'Gemfile', 'composer.json', '.sln',
})

# Max sub-projects to individually index via git ls-files.
# For a parent-of-projects directory (like ~/projects), we only deep-index
# a handful of sub-projects to keep startup fast.  The rest get a shallow
# immediate-files-only listing.
_MAX_SUBPROJECTS = 8


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


class _GitignoreMatcher:
    """Loads and matches .gitignore patterns from the workspace root.

    Uses the ``pathspec`` library for full .gitignore glob support.
    Falls back to a no-op matcher if pathspec is unavailable.
    """

    def __init__(self, workspace: Path):
        self._spec: Optional[Any] = None
        if not HAS_PATHSPEC:
            return
        patterns: List[str] = []
        # Load root .gitignore
        gi = workspace / ".gitignore"
        if gi.is_file():
            try:
                patterns.extend(gi.read_text(errors="replace").splitlines())
            except OSError:
                pass
        # Also load nested .gitignore files at depth 1 (common in monorepos)
        try:
            for entry in os.scandir(workspace):
                if entry.is_dir(follow_symlinks=False):
                    nested_gi = Path(entry.path) / ".gitignore"
                    if nested_gi.is_file():
                        try:
                            prefix = entry.name + "/"
                            for line in nested_gi.read_text(errors="replace").splitlines():
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    # Prefix patterns with the directory they came from
                                    if line.startswith("/"):
                                        patterns.append(prefix + line[1:])
                                    elif line.startswith("!"):
                                        if line[1:].startswith("/"):
                                            patterns.append("!" + prefix + line[2:])
                                        else:
                                            patterns.append("!" + prefix + line[1:])
                                    else:
                                        patterns.append(prefix + line)
                        except OSError:
                            pass
        except OSError:
            pass

        if patterns:
            try:
                self._spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            except Exception as e:
                log.debug("Failed to parse .gitignore patterns: %s", e)

    def is_ignored(self, rel_path: str) -> bool:
        """Return True if the relative path matches a .gitignore pattern."""
        if self._spec is None:
            return False
        return self._spec.match_file(rel_path)


class WorkspaceIndex:
    """Index of files in the workspace, built at startup."""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.files: List[FileInfo] = []
        self._by_dir: Dict[str, List[FileInfo]] = defaultdict(list)
        self._build_time: float = 0.0
        self._is_git: bool = False
        self._is_shallow: bool = False  # True when depth-limited scan was used
        self._dir_count: int = 0  # total directories seen (even if not fully scanned)
        self._skipped_files: int = 0  # files skipped due to depth cap

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
            file_list = self._smart_walk(ws)
            log.info(
                "Index: smart walk (%d files, %d dirs, %d skipped, shallow=%s)",
                len(file_list), self._dir_count, self._skipped_files, self._is_shallow,
            )

        # Decide whether to count lines (expensive for large workspaces)
        count_lines = len(file_list) <= _LINE_COUNT_FILE_CAP

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

            # Count lines for text files only when workspace is small enough
            lines = 0
            if count_lines and not is_binary and size < 2_000_000:
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

    def _smart_walk(self, ws: Path) -> List[str]:
        """Smart filesystem walk with sub-project detection and depth limiting.

        Strategy:
        1. Scan the root directory for immediate children.
        2. For each child directory, check if it's a sub-project (has .git,
           package.json, pyproject.toml, etc.).
        3. Sub-projects with .git get indexed via ``git ls-files`` (fast + accurate).
        4. Other directories get a shallow ``os.scandir``-based walk with strict
           depth limits and per-directory file caps.
        5. .gitignore patterns from the root are respected throughout.
        """
        gi_matcher = _GitignoreMatcher(ws)
        result: List[str] = []
        self._dir_count = 0
        self._skipped_files = 0
        self._is_shallow = False
        subprojects_indexed = 0

        # --- Phase 1: Collect root-level files ---
        try:
            for entry in os.scandir(ws):
                if entry.is_file(follow_symlinks=False):
                    if entry.name.startswith('.'):
                        continue
                    if gi_matcher.is_ignored(entry.name):
                        continue
                    result.append(entry.name)
        except OSError:
            pass
        self._dir_count += 1

        # --- Phase 2: Process each top-level subdirectory ---
        try:
            subdirs = sorted(
                entry.name for entry in os.scandir(ws)
                if entry.is_dir(follow_symlinks=False)
                and entry.name not in SKIP_DIRS
                and not entry.name.startswith('.')
            )
        except OSError:
            subdirs = []

        # Classify subdirectories
        git_subdirs: List[Tuple[str, Path]] = []   # (name, path)
        project_subdirs: List[Tuple[str, Path]] = []
        other_subdirs: List[Tuple[str, Path]] = []

        for subdir_name in subdirs:
            subdir_path = ws / subdir_name

            if gi_matcher.is_ignored(subdir_name + '/'):
                continue

            if (subdir_path / '.git').exists():
                git_subdirs.append((subdir_name, subdir_path))
            elif self._has_project_marker(subdir_path):
                project_subdirs.append((subdir_name, subdir_path))
            else:
                other_subdirs.append((subdir_name, subdir_path))

        # --- Phase 2a: Index git sub-projects in parallel ---
        git_subdirs = git_subdirs[:_MAX_SUBPROJECTS]
        if git_subdirs:
            git_results = self._parallel_git_ls_files(git_subdirs)
            for subdir_name, files in git_results.items():
                if files is not None:
                    for f in files:
                        rel = f"{subdir_name}/{f}"
                        if not gi_matcher.is_ignored(rel.replace(os.sep, '/')):
                            result.append(rel)
                    self._dir_count += 1

        # --- Phase 2b: Depth-limited scan for non-git projects ---
        for subdir_name, subdir_path in project_subdirs:
            self._shallow_scandir(
                ws, subdir_path, subdir_name, gi_matcher, result, depth=1,
            )
            if len(result) > 10_000:
                log.warning("Index walk truncated at 10,000 files")
                self._is_shallow = True
                break

        # --- Phase 2c: Immediate-only listing for non-project dirs ---
        if len(result) <= 10_000:
            for subdir_name, subdir_path in other_subdirs:
                self._is_shallow = True
                self._list_immediate_files(
                    ws, subdir_path, subdir_name, gi_matcher, result,
                )
                if len(result) > 10_000:
                    log.warning("Index walk truncated at 10,000 files")
                    break

        return result

    def _parallel_git_ls_files(
        self, subdirs: List[Tuple[str, Path]]
    ) -> Dict[str, Optional[List[str]]]:
        """Run git ls-files on multiple sub-projects in parallel.

        Returns a dict mapping subdir name -> file list (or None on failure).
        """
        results: Dict[str, Optional[List[str]]] = {}

        def _run_git(name: str, path: Path) -> Tuple[str, Optional[List[str]]]:
            try:
                proc = subprocess.run(
                    ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                    capture_output=True, text=True, cwd=str(path), timeout=10,
                )
                if proc.returncode != 0:
                    return (name, None)
                files = [f.strip() for f in proc.stdout.splitlines() if f.strip()]
                # Skip os.path.isfile — git output is reliable; build()
                # handles vanished files via stat() OSError
                return (name, files)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                return (name, None)

        # Use thread pool — git ls-files is I/O bound
        max_workers = min(8, len(subdirs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_git, name, path): name
                for name, path in subdirs
            }
            for future in concurrent.futures.as_completed(futures):
                name, files = future.result()
                results[name] = files
                log.debug("Sub-project %s: %s files via git",
                           name, len(files) if files else "failed")

        return results

    @staticmethod
    def _has_project_marker(dir_path: Path) -> bool:
        """Check if a directory looks like a project root."""
        try:
            names = {e.name for e in os.scandir(dir_path)}
        except (OSError, PermissionError):
            return False
        return bool(names & _PROJECT_MARKERS)

    def _list_immediate_files(
        self,
        ws: Path,
        dir_path: Path,
        rel_prefix: str,
        gi_matcher: "_GitignoreMatcher",
        result: List[str],
    ) -> None:
        """List only the immediate files in a directory (no recursion).

        Caps at _DEEP_DIR_FILE_CAP and picks representative files if over cap.
        """
        self._dir_count += 1
        try:
            entries = list(os.scandir(dir_path))
        except (OSError, PermissionError):
            return

        dir_files: List[str] = []
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            if entry.is_file(follow_symlinks=False):
                rel_path = f"{rel_prefix}/{entry.name}"
                if not gi_matcher.is_ignored(rel_path.replace(os.sep, '/')):
                    dir_files.append(rel_path)

        if len(dir_files) > _DEEP_DIR_FILE_CAP:
            self._skipped_files += len(dir_files) - _DEEP_DIR_FILE_CAP
            dir_files = self._pick_representative_files(ws, dir_files, _DEEP_DIR_FILE_CAP)

        result.extend(dir_files)

    def _shallow_scandir(
        self,
        ws: Path,
        dir_path: Path,
        rel_prefix: str,
        gi_matcher: "_GitignoreMatcher",
        result: List[str],
        depth: int,
    ) -> None:
        """Recursively scan a directory with os.scandir, respecting depth limits.

        - depth 1 (immediate child of root): full file listing
        - depth > 1: cap files per directory at _DEEP_DIR_FILE_CAP
        - depth >= _MAX_SCAN_DEPTH: stop recursing
        """
        self._dir_count += 1

        if depth >= _MAX_SCAN_DEPTH:
            self._is_shallow = True
            return

        try:
            entries = list(os.scandir(dir_path))
        except (OSError, PermissionError):
            return

        # Collect files in this directory
        dir_files: List[str] = []
        child_dirs: List[tuple] = []  # (name, path, rel)

        for entry in entries:
            if entry.name.startswith('.'):
                continue

            if entry.is_file(follow_symlinks=False):
                rel_path = f"{rel_prefix}/{entry.name}"
                if not gi_matcher.is_ignored(rel_path.replace(os.sep, '/')):
                    dir_files.append(rel_path)

            elif entry.is_dir(follow_symlinks=False):
                if entry.name in SKIP_DIRS:
                    continue
                child_rel = f"{rel_prefix}/{entry.name}"
                if not gi_matcher.is_ignored(child_rel.replace(os.sep, '/') + '/'):
                    child_dirs.append((entry.name, Path(entry.path), child_rel))

        # Apply per-directory file cap at depth > 1
        if depth > 1 and len(dir_files) > _DEEP_DIR_FILE_CAP:
            self._is_shallow = True
            self._skipped_files += len(dir_files) - _DEEP_DIR_FILE_CAP
            dir_files = self._pick_representative_files(ws, dir_files, _DEEP_DIR_FILE_CAP)

        result.extend(dir_files)

        # Recurse into child directories
        for _, child_path, child_rel in sorted(child_dirs):
            if len(result) > 10_000:
                self._is_shallow = True
                return
            self._shallow_scandir(ws, child_path, child_rel, gi_matcher, result, depth + 1)

    def _pick_representative_files(
        self, ws: Path, rel_paths: List[str], cap: int
    ) -> List[str]:
        """Pick the most representative files from a directory.

        Priority:
        1. Source code files (known languages) over config/data files
        2. Larger files (more likely to be substantive) over tiny ones
        3. README/docs files always included if present
        """
        # Categorize
        priority_files: List[str] = []  # READMEs, docs
        source_files: List[tuple] = []  # (size, path) — known language
        other_files: List[tuple] = []   # (size, path) — everything else

        for rel_path in rel_paths:
            fname = os.path.basename(rel_path).lower()
            ext = os.path.splitext(rel_path)[1].lower()

            # Always include READMEs and important config files
            if fname in ('readme.md', 'readme.txt', 'readme.rst', 'readme',
                         'cargo.toml', 'package.json', 'pyproject.toml',
                         'go.mod', 'pom.xml', 'build.gradle', 'makefile',
                         'dockerfile', 'docker-compose.yml'):
                priority_files.append(rel_path)
                continue

            try:
                size = (ws / rel_path).stat().st_size
            except OSError:
                size = 0

            if ext in LANG_MAP:
                source_files.append((size, rel_path))
            else:
                other_files.append((size, rel_path))

        # Sort by size descending (larger = more substantive)
        source_files.sort(key=lambda x: -x[0])
        other_files.sort(key=lambda x: -x[0])

        picked: List[str] = priority_files[:cap]
        remaining = cap - len(picked)

        # Fill with source files first, then other files
        for _, path in source_files:
            if remaining <= 0:
                break
            picked.append(path)
            remaining -= 1

        for _, path in other_files:
            if remaining <= 0:
                break
            picked.append(path)
            remaining -= 1

        return picked

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

        line_info = f"~{total_lines:,} lines" if total_lines > 0 else "lines not counted"
        parts.append(
            f"PROJECT MAP — {len(self.files)} files, "
            f"{line_info} | {lang_summary}"
        )
        if self._is_git:
            parts.append("(git repo — .gitignore respected)")
        elif self._is_shallow:
            skipped_note = f", {self._skipped_files} files skipped" if self._skipped_files else ""
            parts.append(f"(shallow scan — depth-limited{skipped_note})")
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
                    if f.lines > 0:
                        detail = f"{f.lines}L"
                    elif f.size > 0:
                        detail = f"{f.size}B"
                    else:
                        detail = "empty"
                    lang_tag = f" [{f.language}]" if f.language else ""
                    parts.append(f"  {fname} ({detail}{lang_tag})")
                remaining -= 1 + len(dir_files)
            else:
                # Summarise large directories
                text_files = [f for f in dir_files if not f.is_binary]
                dir_lines = sum(f.lines for f in text_files)
                dir_langs = set(f.language for f in dir_files if f.language)
                lang_str = ", ".join(sorted(dir_langs)[:3])
                line_detail = f"~{dir_lines:,}L" if dir_lines > 0 else f"{len(dir_files)} files"
                parts.append(
                    f"{dir_label} — {len(dir_files)} files, "
                    f"{line_detail}"
                    + (f" [{lang_str}]" if lang_str else "")
                )
                # Show top 3 largest files
                top = sorted(text_files, key=lambda f: f.lines or f.size, reverse=True)[:3]
                for f in top:
                    fname = os.path.basename(f.rel_path)
                    if f.lines > 0:
                        parts.append(f"  {fname} ({f.lines}L)")
                    else:
                        parts.append(f"  {fname} ({f.size}B)")
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