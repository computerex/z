"""Scheduled prompts, stored in ``<project>/.claude/scheduled_tasks.json``.

Tasks come in two flavours:

- **One-shot** (``recurring=False``) — fire once, then auto-delete.
- **Recurring** (``recurring=True``) — fire on schedule, reschedule from now,
  persist until explicitly deleted or auto-expired.

File format::

    {"tasks": [{"id": "...", "cron": "...", "prompt": "...", "createdAt": ..., ...}]}
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .cron import next_cron_run_ms, parse_cron_expression

# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class CronTask:
    """A single scheduled task."""

    id: str
    """8-hex-char identifier (first 8 chars of UUID4)."""

    cron: str
    """5-field cron expression in local time."""

    prompt: str
    """Prompt text to enqueue when the task fires."""

    created_at: float
    """Epoch ms when the task was created."""

    last_fired_at: Optional[float] = None
    """Epoch ms of the most recent fire.  Only meaningful for recurring tasks."""

    recurring: bool = False
    """When True the task reschedules after firing instead of being deleted."""

    permanent: bool = False
    """Exempt from auto-expiry.  Not settable via CronCreate — only written
    directly to the JSON file."""

    # ── Runtime-only fields (never serialised) ──────────────────────────
    durable: bool = True
    """``False`` = session-scoped; never written to disk.  Dies with process."""

    agent_id: Optional[str] = None
    """When set, fires are routed to a teammate queue instead of the main REPL."""


# ── Public helpers — file path ───────────────────────────────────────────────

CRON_FILE_REL = Path(".claude", "scheduled_tasks.json")


def get_cron_file_path(project_dir: Optional[str] = None) -> Path:
    """Return the absolute path to the scheduled-tasks JSON file."""
    base = Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
    return base / CRON_FILE_REL


# ── Read / write helpers ─────────────────────────────────────────────────────


def read_cron_tasks(project_dir: Optional[str] = None) -> List[CronTask]:
    """Read and validate ``.claude/scheduled_tasks.json``.

    Returns an empty list if the file is missing, empty, or malformed.
    Tasks with invalid cron strings are silently dropped (logged at DEBUG).
    """
    path = get_cron_file_path(project_dir)
    if not path.exists():
        return []

    try:
        raw = path.read_text("utf-8")
    except Exception:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    tasks_raw = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks_raw, list):
        return []

    out: List[CronTask] = []
    for t in tasks_raw:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        cron = t.get("cron")
        prompt = t.get("prompt")
        created_at = t.get("createdAt")
        if not isinstance(tid, str) or not isinstance(cron, str) or not isinstance(prompt, str):
            continue
        if not isinstance(created_at, (int, float)):
            continue
        if not parse_cron_expression(cron):
            continue

        out.append(CronTask(
            id=tid,
            cron=cron,
            prompt=prompt,
            created_at=float(created_at),
            last_fired_at=float(t["lastFiredAt"]) if t.get("lastFiredAt") is not None else None,
            recurring=bool(t.get("recurring", False)),
            permanent=bool(t.get("permanent", False)),
        ))
    return out


def has_cron_tasks_sync(project_dir: Optional[str] = None) -> bool:
    """Fast sync check — does the cron file have any tasks?"""
    path = get_cron_file_path(project_dir)
    if not path.exists():
        return False
    try:
        raw = path.read_text("utf-8")
    except Exception:
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    tasks = data.get("tasks") if isinstance(data, dict) else None
    return isinstance(tasks, list) and len(tasks) > 0


def write_cron_tasks(tasks: List[CronTask], project_dir: Optional[str] = None) -> None:
    """Overwrite ``.claude/scheduled_tasks.json`` with *tasks*.

    Creates ``.claude/`` if missing.  An empty task list writes an empty file
    (rather than deleting) so a file watcher sees a change event.
    """
    path = get_cron_file_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Strip runtime-only fields before serialising
    serializable: List[Dict] = []
    for t in tasks:
        entry = {
            "id": t.id,
            "cron": t.cron,
            "prompt": t.prompt,
            "createdAt": t.created_at,
        }
        if t.last_fired_at is not None:
            entry["lastFiredAt"] = t.last_fired_at
        if t.recurring:
            entry["recurring"] = True
        if t.permanent:
            entry["permanent"] = True
        serializable.append(entry)

    path.write_text(
        json.dumps({"tasks": serializable}, indent=2, ensure_ascii=False) + "\n",
        "utf-8",
    )


# ── Task management ──────────────────────────────────────────────────────────

_MAX_CRON_JOBS = 50

# In-memory session-only tasks.  Not persisted to disk.
_session_tasks: List[CronTask] = []


def _next_id() -> str:
    return uuid.uuid4().hex[:8]


def add_cron_task(
    cron: str,
    prompt: str,
    recurring: bool = False,
    durable: bool = True,
    agent_id: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> str:
    """Append a task and return its generated ID.

    When *durable* is ``False`` the task is held in process memory only
    (dies when the harness process exits).

    The caller is responsible for having already validated the cron string.
    """
    tid = _next_id()
    task = CronTask(
        id=tid,
        cron=cron,
        prompt=prompt,
        created_at=time.time() * 1000,
        recurring=recurring,
        durable=durable,
        agent_id=agent_id,
    )

    if not durable:
        _session_tasks.append(task)
    else:
        tasks = read_cron_tasks(project_dir)
        tasks.append(task)
        write_cron_tasks(tasks, project_dir)

    return tid


def remove_cron_tasks(ids: List[str], project_dir: Optional[str] = None) -> None:
    """Remove tasks by ID.  No-op if none match.

    Sweeps the in-memory session store first, then the file.
    """
    if not ids:
        return
    id_set = set(ids)

    # Sweep session store first
    global _session_tasks
    before = len(_session_tasks)
    _session_tasks = [t for t in _session_tasks if t.id not in id_set]
    removed_session = before - len(_session_tasks)

    # If every ID was accounted for in session tasks, skip the file
    if removed_session == len(ids):
        return

    # Sweep file
    tasks = read_cron_tasks(project_dir)
    remaining = [t for t in tasks if t.id not in id_set]
    if len(remaining) == len(tasks):
        return
    write_cron_tasks(remaining, project_dir)


def mark_cron_tasks_fired(ids: List[str], fired_at: float, project_dir: Optional[str] = None) -> None:
    """Stamp *last_fired_at* on the given recurring tasks and write back.

    Batched so N fires in one scheduler tick = one read-modify-write.
    Only touches file-backed tasks — session tasks die with the process.
    """
    if not ids:
        return
    id_set = set(ids)

    # Check session tasks first (no persistence needed)
    for t in _session_tasks:
        if t.id in id_set and t.recurring:
            t.last_fired_at = fired_at

    # File tasks
    tasks = read_cron_tasks(project_dir)
    changed = False
    for t in tasks:
        if t.id in id_set:
            t.last_fired_at = fired_at
            changed = True
    if changed:
        write_cron_tasks(tasks, project_dir)


def list_all_cron_tasks(project_dir: Optional[str] = None) -> List[CronTask]:
    """Return file-backed tasks + session-only tasks merged.

    Session tasks get ``durable=False`` so callers can distinguish them.
    """
    file_tasks = read_cron_tasks(project_dir)
    result = list(file_tasks)
    for t in _session_tasks:
        # Make a shallow copy with durable=False
        copy = CronTask(
            id=t.id, cron=t.cron, prompt=t.prompt, created_at=t.created_at,
            last_fired_at=t.last_fired_at, recurring=t.recurring,
            permanent=t.permanent, durable=False, agent_id=t.agent_id,
        )
        result.append(copy)
    return result


# ── Missed-task detection ────────────────────────────────────────────────────# ── Missed-task detection ────────────────────────────────────────────────────


def find_missed_tasks(tasks: List[CronTask], now_ms: float) -> List[CronTask]:
    """Return tasks whose next scheduled run (computed from *created_at*) is
    in the past.  Used to surface catch-up prompts at startup.
    """
    missed: List[CronTask] = []
    for t in tasks:
        next_ms = next_cron_run_ms(t.cron, t.created_at)
        if next_ms is not None and next_ms < now_ms:
            missed.append(t)
    return missed


_MISSED_TASK_PREFIX = (
    "The following one-shot scheduled {noun} {verb} missed while the "
    "harness was not running. {pronoun} {have} already been removed from "
    ".claude/scheduled_tasks.json.\n\n"
    "Do NOT execute {these_prompts} yet. "
    "First ask the user whether to run {each_one} now. "
    "Only execute if the user confirms."
)


def build_missed_task_notification(missed: List['CronTask']) -> str:
    """Build the notification text for missed one-shot tasks.

    Guidance precedes the task list so a multi-line imperative prompt is
    not interpreted as immediate instructions — the model must ask the
    user first.
    """
    if len(missed) == 1:
        t = missed[0]
        header = _MISSED_TASK_PREFIX.format(
            noun="task was", verb="",
            pronoun="It", have="has",
            these_prompts="this prompt", each_one="it",
        )
        from .cron import cron_to_human
        meta = f"[{cron_to_human(t.cron)}, created at {t.created_at}]"
        # Fence at 3+ backticks to prevent premature close
        longest_run = max((len(m) for m in __import__('re').findall(r'`+', t.prompt)), default=0)
        fence = '`' * max(3, longest_run + 1)
        return f"{header}\n\n{meta}\n{fence}\n{t.prompt}\n{fence}"

    header = _MISSED_TASK_PREFIX.format(
        noun="tasks were", verb="",
        pronoun="They", have="have",
        these_prompts="these prompts", each_one="each one",
    )
    blocks = []
    from .cron import cron_to_human
    for t in missed:
        meta = f"[{cron_to_human(t.cron)}, created at {t.created_at}]"
        longest_run = max((len(m) for m in __import__('re').findall(r'`+', t.prompt)), default=0)
        fence = '`' * max(3, longest_run + 1)
        blocks.append(f"{meta}\n{fence}\n{t.prompt}\n{fence}")
    return f"{header}\n\n" + "\n\n".join(blocks)


# ── Session-task lifecycle ────────────────────────────────────────────────────


def clear_session_tasks() -> None:
    """Remove all in-memory session-only tasks."""
    global _session_tasks
    _session_tasks = []


# ── Jitter config (simplified — no GrowthBook) ────────────────────────────────# ── Jitter config (simplified — no GrowthBook) ────────────────────────────────
# Jitter intentionally spreads firing times to avoid thundering-herd inference
# spikes when many users pick the same cron expression (e.g. "0 * * * *").


@dataclass
class CronJitterConfig:
    """Tuning knobs for the deterministic jitter applied to fire times."""

    recurring_frac: float = 0.1
    """Recurring-task forward delay as a fraction of the interval."""

    recurring_cap_ms: int = 900_000
    """Upper bound on recurring forward delay (15 min)."""

    one_shot_max_ms: int = 90_000
    """One-shot backward lead: max ms a task may fire early."""

    one_shot_floor_ms: int = 0
    """One-shot backward lead floor — guarantees no task fires on the exact
    wall-clock mark when set > 0."""

    one_shot_minute_mod: int = 30
    """Jitter fires landing on minutes where ``minute % N == 0``."""

    recurring_max_age_ms: int = 7 * 24 * 60 * 60 * 1000
    """Recurring tasks auto-expire this many ms after creation (7 days).
    ``0`` = unlimited."""


DEFAULT_CRON_JITTER_CONFIG = CronJitterConfig()


def _jitter_frac(task_id: str) -> float:
    """Deterministic float in [0, 1) derived from the 8-char hex task ID."""
    try:
        return int(task_id[:8], 16) / 0x1_0000_0000
    except ValueError:
        return 0.0


def jittered_next_run_ms(
    cron: str, from_ms: float, task_id: str, cfg: CronJitterConfig = DEFAULT_CRON_JITTER_CONFIG,
) -> Optional[float]:
    """Next fire time with per-task forward jitter.

    Spreads recurring tasks across a window rather than all hitting at the
    exact wall-clock mark.  The delay is proportional to the gap between
    fires (``recurring_frac``, capped at ``recurring_cap_ms``).
    """
    t1 = next_cron_run_ms(cron, from_ms)
    if t1 is None:
        return None
    t2 = next_cron_run_ms(cron, t1 + 1)
    if t2 is None:
        return t1  # Pinned date, no herd risk
    jitter = min(
        _jitter_frac(task_id) * cfg.recurring_frac * (t2 - t1),
        cfg.recurring_cap_ms,
    )
    return t1 + jitter


def one_shot_jittered_next_run_ms(
    cron: str, from_ms: float, task_id: str, cfg: CronJitterConfig = DEFAULT_CRON_JITTER_CONFIG,
) -> Optional[float]:
    """Next fire time with per-task backward jitter.

    One-shot tasks that land on a ``:00`` or ``:30`` minute boundary fire
    slightly early (up to ``one_shot_max_ms``) so not every user's "3pm"
    arrives at the same API instant.
    """
    t1 = next_cron_run_ms(cron, from_ms)
    if t1 is None:
        return None
    # Only jitter when the minute lands on a round number
    from datetime import datetime
    if datetime.fromtimestamp(t1 / 1000).minute % cfg.one_shot_minute_mod != 0:
        return t1
    lead = cfg.one_shot_floor_ms + _jitter_frac(task_id) * (cfg.one_shot_max_ms - cfg.one_shot_floor_ms)
    return max(t1 - lead, from_ms)
