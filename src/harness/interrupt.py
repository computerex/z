"""Keyboard interrupt handling: double-tap Escape to cancel.

Design principles:
- Double-tap Escape (two presses within ESC_WINDOW seconds) = soft interrupt.
- Ctrl+B = send current command to background.
- Ctrl+C (SIGINT) is NOT intercepted here -- Python default handler
  raises KeyboardInterrupt, which the main loop already catches gracefully.
  No custom SIGINT monkey-patching, no phantom-SIGINT guards, no recursion risk.

Windows implementation: msvcrt byte polling.
  - Extended keys (arrows, F-keys) produce a 0x00 or 0xE0 prefix byte followed
    by a scan code -- both bytes are consumed so they never look like Escape.
  - VT multi-byte sequences starting with 0x1b are detected by checking whether
    more bytes arrive within 20ms; if so, drain and discard.
Unix implementation: select + termios cbreak, same VT-sequence drain logic.
"""

import sys
import time
import threading
from typing import Optional, Tuple

# -- State --------------------------------------------------------------------

_lock = threading.Lock()
_interrupted: bool = False
_background: bool = False
_interrupt_reason: str = ""
_last_esc: float = 0.0        # monotonic timestamp of last Escape press
ESC_WINDOW: float = 2.0       # seconds between taps to count as double-tap

# -- Public API ---------------------------------------------------------------

def is_interrupted() -> bool:
    return _interrupted


def is_background_requested() -> bool:
    return _background


def reset_interrupt() -> None:
    global _interrupted, _background, _interrupt_reason, _last_esc
    with _lock:
        _interrupted = False
        _background = False
        _interrupt_reason = ""
        _last_esc = 0.0


def reset_background() -> None:
    global _background
    with _lock:
        _background = False


def trigger_interrupt(reason: str = "user") -> None:
    global _interrupted, _interrupt_reason
    with _lock:
        _interrupted = True
        _interrupt_reason = reason


class _State:
    """Compatibility shim so callers can do get_interrupt_state().snapshot()."""
    def snapshot(self) -> Tuple[bool, str]:
        with _lock:
            return _interrupted, _interrupt_reason


def get_interrupt_state() -> _State:
    return _State()


# -- Monitor thread -----------------------------------------------------------

_stop_event = threading.Event()
_monitor_thread: Optional[threading.Thread] = None


def _on_escape() -> None:
    """Handle one Escape keypress. Triggers interrupt on the second tap."""
    global _interrupted, _interrupt_reason, _last_esc
    now = time.monotonic()
    with _lock:
        if now - _last_esc <= ESC_WINDOW:
            _interrupted = True
            _interrupt_reason = "escape"
            return
        _last_esc = now
    # First tap -- print hint outside the lock.
    try:
        sys.stderr.write("\n  Press Esc again to interrupt\n")
        sys.stderr.flush()
    except Exception:
        pass


def _monitor_windows() -> None:
    """Poll for keypresses on Windows using msvcrt."""
    import msvcrt

    while not _stop_event.is_set():
        if not msvcrt.kbhit():
            _stop_event.wait(0.02)
            continue

        byte = msvcrt.getch()

        if byte in (b'\x00', b'\xe0'):
            # Extended key prefix (arrows, F-keys, Delete, Insert, ...).
            # Consume the trailing scan-code byte so it cannot be misread.
            if msvcrt.kbhit():
                msvcrt.getch()

        elif byte == b'\x1b':
            # Could be a real Escape key or the start of a VT escape sequence
            # (e.g. \x1b[A = cursor-up in VT mode).  Wait 20 ms; if more bytes
            # arrive it is a sequence -- drain and ignore.
            _stop_event.wait(0.02)
            if msvcrt.kbhit():
                while msvcrt.kbhit():
                    msvcrt.getch()
            else:
                _on_escape()

        elif byte == b'\x02':  # Ctrl+B -- background
            global _background
            with _lock:
                _background = True


def _monitor_unix() -> None:
    """Poll for keypresses on Unix using select + termios cbreak."""
    import select
    import termios
    import tty

    old = None
    try:
        old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        while not _stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # Check whether a VT sequence follows within 20 ms.
                r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r2:
                    while select.select([sys.stdin], [], [], 0.0)[0]:
                        sys.stdin.read(1)
                else:
                    _on_escape()
            elif ch == '\x02':  # Ctrl+B
                global _background
                with _lock:
                    _background = True
    except Exception:
        pass
    finally:
        if old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            except Exception:
                pass


def start_monitoring() -> None:
    """Start the keyboard monitor (idempotent, resets interrupt state)."""
    global _monitor_thread
    reset_interrupt()
    _stop_event.clear()
    if _monitor_thread and _monitor_thread.is_alive():
        return
    target = _monitor_windows if sys.platform == "win32" else _monitor_unix
    _monitor_thread = threading.Thread(target=target, daemon=True, name="harness-kbd")
    _monitor_thread.start()


def stop_monitoring() -> None:
    """Stop the keyboard monitor and wait for the thread to exit."""
    global _monitor_thread
    _stop_event.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=0.5)
        _monitor_thread = None
