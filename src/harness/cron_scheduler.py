"""Non-asyncio scheduler core for ``.claude/scheduled_tasks.json``.

Lifecycle
---------
1. ``start()`` → load tasks, acquire scheduler lock, start file watcher +
   periodic check timer.
2. ``check()`` called every 1 s — computes next-fire times, fires eligible
   tasks, persists recurring-task metadata.
3. ``stop()`` → tear down timer, watcher, and release lock.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set

from .cron import cron_to_human
from .cron_tasks import (
    CronJitterConfig,
    CronTask,
    DEFAULT_CRON_JITTER_CONFIG,
    build_missed_task_notification,
    find_missed_tasks,
    get_cron_file_path,
    has_cron_tasks_sync,
    jittered_next_run_ms,
    mark_cron_tasks_fired,
    one_shot_jittered_next_run_ms,
    read_cron_tasks,
    remove_cron_tasks,
)
from .cron_tasks_lock import release_scheduler_lock, try_acquire_scheduler_lock

log = logging.getLogger("harness.cron")

CHECK_INTERVAL_S = 1.0          # seconds between check() ticks
FILE_POLL_INTERVAL_S = 3.0      # fallback poll when watchdog is absent
LOCK_PROBE_INTERVAL_S = 5.0     # non-owner re-probe interval


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CronSchedulerOptions:
    """Configuration for ``CronScheduler``."""

    on_fire: Callable[[str], None]
    """Called with the prompt text when a task fires (unless *on_fire_task* is
    given, in which case *on_fire* is NOT called for that fire)."""

    on_fire_task: Optional[Callable[[CronTask], None]] = None
    """When provided, receives the full ``CronTask`` on normal fires instead
    of just the prompt string."""

    on_missed: Optional[Callable[[List[CronTask]], None]] = None
    """Called on initial load with missed one-shot tasks.  When omitted,
    missed tasks are surfaced via *on_fire* with a pre-formatted
    notification."""

    project_dir: Optional[str] = None
    """Explicit project directory.  Defaults to ``cwd``."""

    lock_identity: Optional[str] = None
    """Stable identity for the scheduler lock file.  Defaults to ``pid:<PID>``."""

    get_jitter_config: Optional[Callable[[], CronJitterConfig]] = None
    """Called once per check() tick to get the current jitter config.
    Defaults to ``DEFAULT_CRON_JITTER_CONFIG``."""

    is_killed: Optional[Callable[[], bool]] = None
    """Polled once per check() tick.  When ``True``, check() bails early."""

    filter: Optional[Callable[[CronTask], bool]] = None
    """Per-task gate.  Tasks returning ``False`` are invisible to this
    scheduler — never fired, never stamped, never surfaced as missed."""


class CronScheduler:
    """Periodic scheduler for cron tasks.

    Not thread-safe — all timer-driven callbacks run on a single background
    thread, but ``start()`` / ``stop()`` must be called from the **same**
    thread (typically the main thread).
    """

    def __init__(self, options: CronSchedulerOptions) -> None:
        self._opts = options
        self._dir = options.project_dir

        # File-backed tasks (reloaded on file change)
        self._tasks: List[CronTask] = []
        # Per-task next-fire times (epoch ms)
        self._next_fire: dict[str, float] = {}
        # IDs already enqueued as "missed" — prevents re-asking
        self._missed_asked: Set[str] = set()
        # IDs currently enqueued but not yet cleaned up
        self._in_flight: Set[str] = set()

        self._is_owner = False
        self._started = False
        self._stop_event = threading.Event()

        # Timers (created in start())
        self._check_timer: Optional[threading.Timer] = None
        self._file_poll_timer: Optional[threading.Timer] = None
        self._lock_probe_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler.  Idempotent."""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()

        log.info("CronScheduler starting (dir=%s)", self._dir or os.getcwd())

        # Acquire the scheduler lock
        self._is_owner = try_acquire_scheduler_lock(self._dir, self._opts.lock_identity)
        if not self._is_owner:
            log.info("Did not acquire scheduler lock — will probe periodically")
            self._schedule_lock_probe()
        else:
            log.info("Acquired scheduler lock")

        # Initial load
        self._load(initial=True)

        # Start the periodic file poll (avoids watchdog dependency)
        self._schedule_file_poll()

        # Start the check timer
        self._schedule_check()

    def stop(self) -> None:
        """Stop the scheduler and release resources."""
        if not self._started:
            return
        self._started = False
        self._stop_event.set()

        # Cancel timers
        for t in (self._check_timer, self._file_poll_timer, self._lock_probe_timer):
            if t is not None:
                t.cancel()

        if self._is_owner:
            release_scheduler_lock(self._dir)
            self._is_owner = False

        log.info("CronScheduler stopped")

    def get_next_fire_time(self) -> Optional[float]:
        """Epoch ms of the soonest scheduled fire, or ``None``."""
        with self._lock:
            best = float("inf")
            for v in self._next_fire.values():
                if v < best:
                    best = v
        return best if best != float("inf") else None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _schedule_check(self) -> None:
        if self._stop_event.is_set():
            return
        self._check_timer = threading.Timer(CHECK_INTERVAL_S, self._check)
        self._check_timer.daemon = True
        self._check_timer.start()

    def _schedule_file_poll(self) -> None:
        if self._stop_event.is_set():
            return
        self._file_poll_timer = threading.Timer(FILE_POLL_INTERVAL_S, self._poll_file)
        self._file_poll_timer.daemon = True
        self._file_poll_timer.start()

    def _schedule_lock_probe(self) -> None:
        if self._stop_event.is_set():
            return
        self._lock_probe_timer = threading.Timer(LOCK_PROBE_INTERVAL_S, self._probe_lock)
        self._lock_probe_timer.daemon = True
        self._lock_probe_timer.start()

    # ── Lock management ──────────────────────────────────────────────────────

    def _probe_lock(self) -> None:
        if self._stop_event.is_set() or self._is_owner:
            return
        if try_acquire_scheduler_lock(self._dir, self._opts.lock_identity):
            self._is_owner = True
            log.info("Acquired scheduler lock via probe")
            # Re-load file tasks now that we own the lock
            self._tasks = read_cron_tasks(self._dir)
            return
        self._schedule_lock_probe()

    # ── File watching (polling) ──────────────────────────────────────────────

    def _poll_file(self) -> None:
        if self._stop_event.is_set():
            return
        self._schedule_file_poll()  # re-arm first
        # Stat the file to detect changes
        path = get_cron_file_path(self._dir)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        # Compare with stored mtime
        prev = getattr(self, "_file_mtime", None)
        if prev is not None and mtime == prev:
            return
        self._file_mtime = mtime
        # Reload
        self._tasks = read_cron_tasks(self._dir)

    # ── Initial load ─────────────────────────────────────────────────────────

    def _load(self, initial: bool) -> None:
        with self._lock:
            self._tasks = read_cron_tasks(self._dir)

            if not initial:
                return

            # Surface missed one-shot tasks on initial load
            now = time.time() * 1000
            missed = find_missed_tasks(self._tasks, now)
            missed = [t for t in missed if not t.recurring and t.id not in self._missed_asked
                      and (self._opts.filter is None or self._opts.filter(t))]
            if not missed:
                return

            for t in missed:
                self._missed_asked.add(t.id)
                self._next_fire[t.id] = float("inf")

            log.info("Found %d missed one-shot task(s)", len(missed))
            if self._opts.on_missed:
                self._opts.on_missed(missed)
            else:
                self._opts.on_fire(build_missed_task_notification(missed))

            # Remove missed tasks from file
            remove_cron_tasks([t.id for t in missed], self._dir)

    # ── Periodic check ───────────────────────────────────────────────────────

    def _check(self) -> None:
        if self._stop_event.is_set():
            return

        # Kill switch
        if self._opts.is_killed is not None and self._opts.is_killed():
            self._schedule_check()
            return

        now = time.time() * 1000
        cfg = self._opts.get_jitter_config() if self._opts.get_jitter_config else DEFAULT_CRON_JITTER_CONFIG
        fired_file_recurring: List[str] = []
        seen: Set[str] = set()

        def process(t: CronTask, is_session: bool) -> None:
            if self._opts.filter is not None and not self._opts.filter(t):
                return
            seen.add(t.id)
            if t.id in self._in_flight:
                return

            next_ms = self._next_fire.get(t.id)
            if next_ms is None:
                # First sight — compute from lastFiredAt or createdAt
                if t.recurring:
                    next_ms = jittered_next_run_ms(
                        t.cron, t.last_fired_at or t.created_at, t.id, cfg,
                    ) or float("inf")
                else:
                    next_ms = one_shot_jittered_next_run_ms(
                        t.cron, t.created_at, t.id, cfg,
                    ) or float("inf")
                self._next_fire[t.id] = next_ms

            if now < next_ms:
                return

            # ── Fire! ──
            log.info("Cron task %s firing (recurring=%s)", t.id, t.recurring)
            if self._opts.on_fire_task:
                self._opts.on_fire_task(t)
            else:
                self._opts.on_fire(t.prompt)

            # Check if recurring task is aged out
            aged = _is_recurring_aged(t, now, cfg.recurring_max_age_ms)

            if t.recurring and not aged:
                # Recurring: reschedule from now
                new_next = jittered_next_run_ms(t.cron, now, t.id, cfg) or float("inf")
                self._next_fire[t.id] = new_next
                if not is_session:
                    fired_file_recurring.append(t.id)
            elif is_session:
                # One-shot (or aged) session task: remove from memory
                from .cron_tasks import remove_cron_tasks as _rct
                _rct([t.id], self._dir)
                self._next_fire.pop(t.id, None)
            else:
                # One-shot (or aged) file task: delete from disk
                self._in_flight.add(t.id)
                remove_cron_tasks([t.id], self._dir)
                self._next_fire.pop(t.id, None)
                # Remove from in-memory task list to prevent refire
                # on the next check cycle (next_fire is gone, so the
                # scheduler would recompute from the cron expr and
                # fire again since the original time is in the past).
                self._tasks = [t2 for t2 in self._tasks if t2.id != t.id]
                self._in_flight.discard(t.id)

        # File-backed tasks (only when we own the lock)
        if self._is_owner:
            with self._lock:
                for t in self._tasks:
                    process(t, False)

            if fired_file_recurring:
                for tid in fired_file_recurring:
                    self._in_flight.add(tid)
                mark_cron_tasks_fired(fired_file_recurring, now, self._dir)
                for tid in fired_file_recurring:
                    self._in_flight.discard(tid)

        # Session-only tasks (always, no lock needed)
        from .cron_tasks import _session_tasks
        for t in _session_tasks:
            process(t, True)

        # Evict stale schedule entries
        if seen:
            stale = [k for k in self._next_fire if k not in seen]
            for k in stale:
                del self._next_fire[k]
        else:
            self._next_fire.clear()

        # Re-arm
        self._schedule_check()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_recurring_aged(t: CronTask, now_ms: float, max_age_ms: int) -> bool:
    """``True`` when a recurring task is past its auto-expiry limit."""
    if max_age_ms == 0:
        return False
    return bool(t.recurring and not t.permanent and now_ms - t.created_at >= max_age_ms)
