"""Interrupt handling — Ctrl+C detection, background request flag, monitoring thread."""

import sys
import threading
import signal
from typing import Optional
from dataclasses import dataclass

from .logger import get_logger

_log = get_logger("interrupt")


@dataclass
class InterruptState:
    """Shared state for interrupt handling."""
    interrupted: bool = False
    background: bool = False
    reason: str = ""
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self):
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.interrupted = False
            self.background = False
            self.reason = ""
    
    def trigger(self, reason: str = "user"):
        with self._lock:
            self.interrupted = True
            self.reason = reason
        _log.info("INTERRUPT triggered: reason=%s", reason)
    
    def trigger_background(self):
        with self._lock:
            self.background = True
            self.reason = "background"
        _log.info("BACKGROUND requested")

    def snapshot(self):
        """Return (interrupted, reason) atomically."""
        with self._lock:
            return self.interrupted, self.reason


# Global interrupt state
_interrupt_state = InterruptState()

# Hard-exit request: set by the second Ctrl+C during a parent turn. The agent
# loop checks this at iteration boundaries and unwinds so the REPL can save
# sessions and exit gracefully (instead of the process dying mid-turn, which
# orphaned in-flight sub-agents and lost the parent's turn state).
_hard_exit_requested = False


def request_hard_exit() -> None:
    """Request a graceful exit at the next agent-loop iteration boundary."""
    global _hard_exit_requested
    _hard_exit_requested = True
    _log.warning("Hard exit requested — unwinding at next iteration boundary")


def is_hard_exit_requested() -> bool:
    """Whether a graceful exit was requested (checked by the agent loop/REPL)."""
    return _hard_exit_requested


def clear_hard_exit() -> None:
    """Clear the hard-exit request (e.g. at REPL prompt after a saved exit)."""
    global _hard_exit_requested
    _hard_exit_requested = False


def get_interrupt_state() -> InterruptState:
    """Get the global interrupt state."""
    return _interrupt_state


def reset_interrupt():
    """Reset interrupt state."""
    _interrupt_state.reset()


def is_interrupted() -> bool:
    """Check if interrupted."""
    return _interrupt_state.interrupted


def is_background_requested() -> bool:
    """Check if background was requested."""
    return _interrupt_state.background


def reset_background():
    """Reset only the background flag (used after sending command to background)."""
    with _interrupt_state._lock:
        _interrupt_state.background = False


class KeyboardMonitor:
    """Monitor for escape key and Ctrl+B during streaming.

    Uses a SINGLE persistent daemon thread (started on first enable) that
    stays alive for the lifetime of the process.  Callers enable/disable
    monitoring via the ``enable()`` / ``disable()`` methods rather than
    destroying and recreating threads — this eliminates the zombie-thread
    race that caused Escape key events to be silently consumed and lost.
    """

    def __init__(self):
        self._enabled = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sigint_installed = False  # SIGINT handler installed at most once

    def enable(self):
        """Enable monitoring for keys.  Resets any stale interrupt state."""
        _log.debug("KeyboardMonitor.enable()")
        reset_interrupt()
        self._enabled.set()
        if self._thread is None:
            self._install_sigint()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def disable(self):
        """Disable monitoring.  The persistent thread stays alive but idle."""
        _log.debug("KeyboardMonitor.disable()")
        self._enabled.clear()

    # ── Signal handling ────────────────────────────────────────────────

    def _install_sigint(self):
        """Install SIGINT handler (once).  Never uninstalled — the monitor
        is a singleton that lives for the whole process lifetime."""
        if self._sigint_installed:
            return
        self._orig_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        self._sigint_installed = True

    def _on_sigint(self, signum, frame):
        """First Ctrl+C → soft interrupt. Second Ctrl+C → request graceful
        unwind (the agent loop saves state at the next iteration boundary)
        instead of hard-killing the process mid-turn."""
        was_already = _interrupt_state.interrupted
        _interrupt_state.trigger("ctrl-c")
        if was_already:
            _log.warning("Second Ctrl+C — requesting graceful exit")
            request_hard_exit()
            # Wake any loop waiting on the interrupt state; the agent loop
            # observes hard-exit at its next boundary and unwinds. Do NOT
            # re-raise here: killing the process mid-turn orphans sub-agents
            # and loses session state.

    # ── Monitor loop (persistent) ──────────────────────────────────────

    def _monitor_loop(self):
        """Persistent top-level monitor loop — dispatches to platform handler."""
        if sys.platform == "win32":
            try:
                self._monitor_windows()
            except Exception as e:
                _log.warning("Win32 console API monitor failed (%s), falling back to msvcrt", e)
                self._monitor_windows_msvcrt()
        else:
            self._monitor_unix()

    # ── Windows: Win32 Console API (primary) ──────────────────────────

    def _monitor_windows(self):
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if handle is None or handle == -1:
            raise RuntimeError("Failed to get console input handle")

        KEY_EVENT = 0x0001
        VK_ESCAPE = 0x1B
        CTRL_MASK = 0x0008 | 0x0004

        class CHAR_UNION(ctypes.Union):
            _fields_ = [("UnicodeChar", wintypes.WCHAR), ("AsciiChar", ctypes.c_char)]

        class KEY_EVENT_RECORD(ctypes.Structure):
            _fields_ = [
                ("bKeyDown", wintypes.BOOL),
                ("wRepeatCount", wintypes.WORD),
                ("wVirtualKeyCode", wintypes.WORD),
                ("wVirtualScanCode", wintypes.WORD),
                ("uChar", CHAR_UNION),
                ("dwControlKeyState", wintypes.DWORD),
            ]

        class INPUT_RECORD(ctypes.Structure):
            class _EVENT(ctypes.Union):
                _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("_pad", ctypes.c_byte * 16)]
            _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]

        _log.info("Win32 console API monitor started (handle=%s)", handle)

        buf = INPUT_RECORD()
        n = wintypes.DWORD(0)

        # ── Persistent loop — single thread for the entire process lifetime ──
        while True:
            if not kernel32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(n)) or n.value == 0:
                threading.Event().wait(0.02)
                continue

            if not kernel32.ReadConsoleInputW(handle, ctypes.byref(buf), 1, ctypes.byref(n)) or n.value == 0:
                threading.Event().wait(0.02)
                continue

            # When disabled, drain events silently (don't act, don't pile up)
            if not self._enabled.is_set():
                continue

            if buf.EventType != KEY_EVENT:
                continue
            ke = buf.Event.KeyEvent
            if not ke.bKeyDown:
                continue

            vk = ke.wVirtualKeyCode
            scan = ke.wVirtualScanCode
            ctrl = bool(ke.dwControlKeyState & CTRL_MASK)

            # Real keys have non-zero scan code — reject injected VT sequences
            if scan == 0:
                continue

            if vk == VK_ESCAPE:
                _interrupt_state.trigger("escape")
            elif ctrl and vk == 0x42:  # Ctrl+B
                _interrupt_state.trigger_background()

    # ── Windows: msvcrt fallback (Ctrl+B only) ─────────────────────────

    def _monitor_windows_msvcrt(self):
        import msvcrt
        while True:
            if not self._enabled.is_set():
                threading.Event().wait(0.02)
                continue
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b"\x02":  # Ctrl+B
                    _interrupt_state.trigger_background()
            threading.Event().wait(0.02)

    # ── Unix ───────────────────────────────────────────────────────────

    def _monitor_unix(self):
        import select
        import termios
        import tty

        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if not self._enabled.is_set():
                    threading.Event().wait(0.02)
                    continue
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    key = sys.stdin.read(1)
                    if key == "\x1b":  # Escape
                        _interrupt_state.trigger("escape")
                    elif key == "\x02":  # Ctrl+B
                        _interrupt_state.trigger_background()
        except Exception:
            pass
        finally:
            if old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# Singleton monitor
_monitor: Optional[KeyboardMonitor] = None


def get_monitor() -> KeyboardMonitor:
    """Get or create the keyboard monitor."""
    global _monitor
    if _monitor is None:
        _monitor = KeyboardMonitor()
    return _monitor


def start_monitoring():
    """Enable keyboard monitoring (persistent thread)."""
    get_monitor().enable()


def stop_monitoring():
    """Disable keyboard monitoring (thread stays alive)."""
    get_monitor().disable()
