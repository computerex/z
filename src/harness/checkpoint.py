"""Checkpoint system for undo/redo of agent turns.

Takes lightweight git-based snapshots of the workspace before each agent
turn. Users can undo/redo entire turns (file changes + conversation) with
/undo and /redo (or Ctrl+Z / Ctrl+Y).

Storage: ~/.z/checkpoints/<project_hash>/  (separate git repo per project)
"""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logger import get_logger

log = get_logger("checkpoint")

# ── Binary / non-text patterns for git exclude ───────────────────
# Only text source code and config files should be checkpointed.
# Everything else is excluded so `git add -A` never stages binaries.
_BINARY_EXCLUDE_PATTERNS = """
# ── Executables & shared libraries ──
*.exe
*.dll
*.so
*.dylib
*.o
*.a
*.lib
*.com
*.msi
*.app

# ── Compiled / bytecode ──
*.pyc
*.pyo
*.class
*.jar
*.war
*.ear
*.elc
*.beam
*.wasm

# ── Archives & compressed ──
*.zip
*.tar
*.gz
*.bz2
*.xz
*.7z
*.rar
*.zst
*.lz4
*.cab
*.iso
*.dmg
*.pkg
*.deb
*.rpm

# ── Images ──
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.ico
*.webp
*.tiff
*.tif
*.psd
*.ai
*.eps
*.raw
*.cr2
*.nef
*.heic
*.avif

# ── Audio ──
*.mp3
*.wav
*.ogg
*.flac
*.aac
*.wma
*.m4a
*.opus

# ── Video ──
*.mp4
*.avi
*.mov
*.mkv
*.wmv
*.flv
*.webm
*.m4v
*.mpeg
*.mpg
*.vob

# ── Fonts ──
*.woff
*.woff2
*.ttf
*.otf
*.eot

# ── Documents (binary formats) ──
*.pdf
*.doc
*.docx
*.xls
*.xlsx
*.ppt
*.pptx
*.odt
*.ods
*.odp

# ── Databases ──
*.db
*.sqlite
*.sqlite3
*.mdb
*.accdb
*.ldf
*.mdf

# ── Data / blobs ──
*.bin
*.dat
*.pak
*.vri
*.bak
*.dump
*.img
*.vhd
*.vhdx
*.qcow2
*.vmdk

# ── ML models ──
*.model
*.onnx
*.pt
*.pth
*.h5
*.tflite
*.pb
*.safetensors
*.gguf
*.ggml

# ── Game / media assets ──
*.unity3d
*.unitypackage
*.asset
*.prefab
*.fbx
*.obj
*.blend
*.max
*.3ds
*.dds
*.ktx
*.pvr
*.astc

# ── Node / build artifacts ──
node_modules/
.git/
__pycache__/
.venv/
venv/
env/
.env
build/
dist/
target/
.next/
.nuxt/
.output/
coverage/
.tox/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# ── OS junk ──
Thumbs.db
.DS_Store
desktop.ini
"""


@dataclass
class Checkpoint:
    """A single snapshot of workspace state."""
    tree_hash: str              # git tree object hash
    timestamp: float            # when the checkpoint was taken
    user_input: str             # the user request that triggered this turn
    message_count: int          # len(agent.messages) at checkpoint time
    model: str = ""             # model that was active


@dataclass
class CheckpointDiff:
    """Summary of changes between two checkpoints."""
    files_modified: List[str] = field(default_factory=list)
    files_added: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0


class CheckpointManager:
    """Git-backed workspace checkpointing for undo/redo."""

    def __init__(self, workspace_path: str, max_checkpoints: int = 50):
        self.workspace_path = os.path.abspath(workspace_path)
        self.max_checkpoints = max_checkpoints

        # Separate git repo for snapshots
        project_hash = hashlib.sha256(
            self.workspace_path.encode()
        ).hexdigest()[:16]
        self.git_dir = Path.home() / ".z" / "checkpoints" / project_hash
        self._initialized = False

        # Checkpoint stack: [...older, newer...]
        self.checkpoints: List[Checkpoint] = []
        # Position in the stack (-1 = at head, i.e. no undo active)
        self._undo_pos: int = -1
        # Stashed messages for redo (saved when undoing)
        self._redo_messages: List[list] = []
        # Pre-undo snapshot for redo
        self._redo_tree: Optional[str] = None
        # Whether git is available on this system
        self._git_available: Optional[bool] = None
        # Cached result of workspace size check
        self._too_large: Optional[bool] = None

    # ── Git plumbing ─────────────────────────────────────────────

    def _check_git(self) -> bool:
        """Check if git CLI is available. Cached after first check."""
        if self._git_available is not None:
            return self._git_available
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            self._git_available = True
        except (FileNotFoundError, subprocess.SubprocessError):
            self._git_available = False
            log.warning(
                "git not found — undo/redo disabled. "
                "Install git to enable checkpoint support."
            )
        return self._git_available

    def _git(self, *args, check: bool = True, capture: bool = True,
             use_work_tree: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a git command against the snapshot repo."""
        cmd = [
            "git",
            f"--git-dir={self.git_dir}",
        ]
        if use_work_tree:
            cmd.append(f"--work-tree={self.workspace_path}")
        cmd.extend(args)
        try:
            # Use Popen + communicate for reliable timeout handling.
            # subprocess.run with timeout on Windows can hang if the child
            # process spawns helpers that inherit stdout/stderr handles.
            kwargs = dict(
                text=True,
                cwd=self.workspace_path,
            )
            if capture:
                kwargs["stdout"] = subprocess.PIPE
                kwargs["stderr"] = subprocess.PIPE
            # On Windows, CREATE_NEW_PROCESS_GROUP lets us kill the tree
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(cmd, **kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process tree
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True, timeout=5,
                        )
                    else:
                        proc.kill()
                except Exception:
                    proc.kill()
                proc.wait(timeout=5)
                log.error("Git command timed out after %ds: %s", timeout, " ".join(args[:3]))
                raise subprocess.TimeoutExpired(cmd, timeout)
            if check and proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stdout, stderr,
                )
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            raise
        except subprocess.CalledProcessError as e:
            log.error("Git command failed: %s\nstderr: %s", " ".join(args[:3]), e.stderr)
            raise

    def _ensure_repo(self) -> bool:
        """Initialize the snapshot git repo if needed."""
        if self._initialized:
            return True
        if not self._check_git():
            return False
        try:
            self.git_dir.mkdir(parents=True, exist_ok=True)

            if not (self.git_dir / "HEAD").exists():
                self._git("init", "--bare", use_work_tree=False)
                # Configure for speed and cross-platform
                self._git("config", "core.autocrlf", "false")
                self._git("config", "core.longpaths", "true")
                self._git("config", "core.fsmonitor", "false")
                self._git("config", "gc.auto", "0")
                log.info("Initialized checkpoint repo at %s", self.git_dir)

            # Build exclude file: always include binary patterns, plus project .gitignore
            project_gitignore = Path(self.workspace_path) / ".gitignore"
            exclude_file = self.git_dir / "info" / "exclude"
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            exclude_parts = [_BINARY_EXCLUDE_PATTERNS.strip()]
            if project_gitignore.exists():
                try:
                    exclude_parts.append(
                        "\n# ── Project .gitignore ──\n"
                        + project_gitignore.read_text(errors="replace")
                    )
                except OSError:
                    pass
            exclude_file.write_text("\n".join(exclude_parts) + "\n")

            self._initialized = True
            return True
        except Exception as e:
            log.error("Failed to initialize checkpoint repo: %s", e)
            return False

    # ── Snapshot operations ──────────────────────────────────────

    def _workspace_too_large(self) -> bool:
        """Quick heuristic to skip checkpointing for huge non-code directories.

        First checks the WorkspaceIndex (if available) — if the index found
        no git repo and used a shallow/depth-limited scan, the workspace is
        not a code project and checkpointing is skipped.

        Falls back to a filesystem scan that bails at 500 MB or 500 files.
        """
        if self._too_large is not None:
            return self._too_large

        # Check workspace index if the agent linked one
        idx = getattr(self, '_workspace_index', None)
        if idx is not None:
            if not idx._is_git and idx._is_shallow:
                log.info(
                    "Workspace not a git project (shallow index) — "
                    "skipping checkpointing"
                )
                self._too_large = True
                return True

        # Fallback: quick filesystem scan
        try:
            top = Path(self.workspace_path)
            total_size = 0
            file_count = 0
            MAX_FILES = 500
            MAX_SIZE = 500 * 1024 * 1024  # 500 MB
            for entry in top.rglob("*"):
                if entry.is_file():
                    total_size += entry.stat().st_size
                    file_count += 1
                    if total_size > MAX_SIZE:
                        log.info(
                            "Workspace too large for checkpointing: >%dMB after %d files",
                            MAX_SIZE // (1024 * 1024), file_count,
                        )
                        self._too_large = True
                        return True
                    if file_count > MAX_FILES:
                        break  # Don't spend forever scanning
        except Exception:
            pass
        self._too_large = False
        return False

    def take_snapshot(self, user_input: str, message_count: int, model: str = "") -> Optional[str]:
        """Capture current workspace state. Returns tree hash or None on failure."""
        if not self._ensure_repo():
            return None

        # Skip huge directories (game installs, media libraries, etc.)
        if self._workspace_too_large():
            return None

        try:
            # Stage all files (respecting .gitignore via exclude)
            self._git("add", "-A", timeout=15)

            # Write tree object (lightweight — no commit overhead)
            result = self._git("write-tree")
            tree_hash = result.stdout.strip()

            if not tree_hash:
                log.warning("write-tree returned empty hash")
                return None

            cp = Checkpoint(
                tree_hash=tree_hash,
                timestamp=time.time(),
                user_input=user_input,
                message_count=message_count,
                model=model,
            )

            # If we're in an undo state and taking a new snapshot,
            # discard the redo future (user chose a new path)
            if self._undo_pos >= 0:
                # Trim checkpoints after current undo position
                self.checkpoints = self.checkpoints[:self._undo_pos + 1]
                self._undo_pos = -1
                self._redo_tree = None
                self._redo_messages.clear()

            self.checkpoints.append(cp)

            # Cap history
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints = self.checkpoints[-self.max_checkpoints:]

            log.info(
                "Checkpoint taken: %s (files staged, %d total checkpoints)",
                tree_hash[:12], len(self.checkpoints),
            )
            return tree_hash

        except Exception as e:
            log.error("Failed to take snapshot: %s", e)
            return None

    def get_diff_since(self, tree_hash: str) -> CheckpointDiff:
        """Get a summary of file changes since a given tree hash."""
        diff = CheckpointDiff()
        if not self._ensure_repo():
            return diff

        try:
            # Stage current state first
            self._git("add", "-A")
            current_tree = self._git("write-tree").stdout.strip()

            result = self._git(
                "diff", "--numstat", tree_hash, current_tree,
                check=False,
            )
            if result.returncode != 0:
                return diff

            for line in result.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                added, removed, filepath = parts[0], parts[1], parts[2]
                try:
                    diff.insertions += int(added) if added != "-" else 0
                    diff.deletions += int(removed) if removed != "-" else 0
                except ValueError:
                    pass
                diff.files_modified.append(filepath)

            # Separate added/deleted via diff-filter
            for filter_char, target_list in [("A", diff.files_added), ("D", diff.files_deleted)]:
                result = self._git(
                    "diff", f"--diff-filter={filter_char}", "--name-only",
                    tree_hash, current_tree,
                    check=False,
                )
                if result.returncode == 0:
                    for f in result.stdout.strip().splitlines():
                        if f:
                            target_list.append(f)

        except Exception as e:
            log.error("get_diff_since failed: %s", e)

        return diff

    def restore_snapshot(self, tree_hash: str) -> bool:
        """Restore workspace files to a given tree hash."""
        if not self._ensure_repo():
            return False

        try:
            self._git("read-tree", tree_hash)
            self._git("checkout-index", "-a", "-f")
            log.info("Restored workspace to %s", tree_hash[:12])
            return True
        except Exception as e:
            log.error("Failed to restore snapshot %s: %s", tree_hash[:12], e)
            return False

    # ── Undo / Redo ──────────────────────────────────────────────

    def can_undo(self) -> bool:
        """True if there's a checkpoint to undo to."""
        if len(self.checkpoints) < 2:
            return False
        if self._undo_pos == -1:
            return True  # at head, can go back
        return self._undo_pos > 0

    def can_redo(self) -> bool:
        """True if there's a redo available."""
        if self._undo_pos == -1:
            return False
        return self._undo_pos < len(self.checkpoints) - 1 or self._redo_tree is not None

    def undo(self, current_messages: list) -> Optional[Tuple[Checkpoint, CheckpointDiff, list]]:
        """Undo the last turn. Returns (checkpoint, diff, restored_messages) or None.

        The caller should replace agent.messages with the returned messages.
        """
        if not self.can_undo():
            return None

        if self._undo_pos == -1:
            # First undo — save current state for redo
            try:
                self._git("add", "-A")
                self._redo_tree = self._git("write-tree").stdout.strip()
            except Exception:
                self._redo_tree = None
            self._redo_messages = [current_messages[:]]
            self._undo_pos = len(self.checkpoints) - 1

        # Move back one position
        target_pos = self._undo_pos - 1
        if target_pos < 0:
            return None

        target_cp = self.checkpoints[target_pos]

        # Get diff showing what's being undone
        diff = self.get_diff_since(target_cp.tree_hash)

        # Restore files
        if not self.restore_snapshot(target_cp.tree_hash):
            return None

        # Build restored messages (truncate to checkpoint's message count)
        restored_messages = current_messages[:target_cp.message_count]

        self._undo_pos = target_pos
        log.info("Undo to checkpoint %d (%s)", target_pos, target_cp.tree_hash[:12])

        return target_cp, diff, restored_messages

    def redo(self, current_messages: list) -> Optional[Tuple[Checkpoint, CheckpointDiff, list]]:
        """Redo — move forward one checkpoint. Returns (checkpoint, diff, restored_messages) or None."""
        if not self.can_redo():
            return None

        target_pos = self._undo_pos + 1

        if target_pos < len(self.checkpoints):
            target_cp = self.checkpoints[target_pos]
            diff = self.get_diff_since(target_cp.tree_hash)
            if not self.restore_snapshot(target_cp.tree_hash):
                return None
            self._undo_pos = target_pos

            # If at head and no dirty state to restore, clear undo state
            if target_pos == len(self.checkpoints) - 1 and not self._redo_tree:
                restored_messages = self._redo_messages[0] if self._redo_messages else current_messages
                self._undo_pos = -1
                self._redo_messages.clear()
            else:
                restored_messages = current_messages[:target_cp.message_count]

            log.info("Redo to checkpoint %d (%s)", target_pos, target_cp.tree_hash[:12])
            return target_cp, diff, restored_messages

        # Redo to head (restore original dirty state before any undo)
        if self._redo_tree:
            diff = self.get_diff_since(self._redo_tree)
            if not self.restore_snapshot(self._redo_tree):
                return None
            restored_messages = self._redo_messages[0] if self._redo_messages else current_messages
            fake_cp = Checkpoint(
                tree_hash=self._redo_tree,
                timestamp=time.time(),
                user_input="(current state)",
                message_count=len(restored_messages),
            )
            self._undo_pos = -1
            self._redo_tree = None
            self._redo_messages.clear()
            return fake_cp, diff, restored_messages

        return None

    # ── Display helpers ──────────────────────────────────────────

    def get_history_display(self, current_model: str = "") -> List[dict]:
        """Return checkpoint history for display."""
        items = []
        for i, cp in enumerate(self.checkpoints):
            items.append({
                "index": i,
                "user_input": cp.user_input[:80],
                "model": cp.model,
                "timestamp": cp.timestamp,
                "is_current": (
                    i == self._undo_pos if self._undo_pos >= 0
                    else i == len(self.checkpoints) - 1
                ),
            })
        return items

    def gc(self) -> None:
        """Garbage collect old git objects."""
        if not self._initialized:
            return
        try:
            self._git("gc", "--prune=7.days.ago", "--quiet", check=False)
        except Exception:
            pass
