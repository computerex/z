"""Keyboard interrupt handling for streaming operations."""

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
    _interrupt_state.background = False


def trigger_interrupt(reason: str = "user"):
    """Trigger an interrupt externally."""
    _interrupt_state.trigger(reason)


class KeyboardMonitor:
    """Monitor for escape key and Ctrl+B during streaming."""
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._original_sigint = None
    
    def start(self):
        """Start monitoring for keys."""
        if self._running and self._thread and self._thread.is_alive():
            _log.debug("KeyboardMonitor.start() — already running")
            return
        
        _log.debug("KeyboardMonitor.start()")
        self._running = True
        self._stop_event.clear()
        reset_interrupt()
        
        # Install Ctrl+C handler
        self._original_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def _sigint_handler(self, signum, frame):
        """Handle Ctrl+C by triggering interrupt.
        
        First Ctrl+C → soft interrupt (set flag, agent checks it).
        Second Ctrl+C → hard exit (raise KeyboardInterrupt).
        """
        was_already_interrupted = _interrupt_state.interrupted
        _interrupt_state.trigger("ctrl-c")
        if was_already_interrupted:
            # Second Ctrl+C while already interrupted — hard exit
            _log.warning("Second Ctrl+C — hard exit")
            if self._original_sigint and self._original_sigint != signal.SIG_DFL:
                self._original_sigint(signum, frame)
            else:
                raise KeyboardInterrupt()
    
    def stop(self):
        """Stop monitoring."""
        _log.debug("KeyboardMonitor.stop()")
        self._running = False
        self._stop_event.set()
        
        # Restore original signal handler
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
        
        if self._thread:
            self._thread.join(timeout=0.2)
            self._thread = None
    
    def _monitor_loop(self):
        """Monitor for keys in background thread."""
        if sys.platform == 'win32':
            self._monitor_windows()
        else:
            self._monitor_unix()
    
    def _monitor_windows(self):
        """Windows-specific keyboard monitoring using Win32 Console API.

        Uses ReadConsoleInput to get proper KEY_EVENT_RECORD structures
        with virtual key codes, completely avoiding the msvcrt.kbhit()/getch()
        problem where VT escape sequences, terminal focus events, and other
        injected bytes produce phantom 0x1b that looks like the Escape key.
        """
        try:
            self._monitor_windows_console_api()
        except Exception as e:
            _log.warning("Win32 console API monitor failed (%s), falling back to msvcrt", e)
            self._monitor_windows_msvcrt()

    def _monitor_windows_console_api(self):
        """Primary Windows monitor using Win32 ReadConsoleInput."""
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # Constants
        KEY_EVENT = 0x0001
        VK_ESCAPE = 0x1B
        CTRL_MASK = 0x0008 | 0x0004  # LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED

        # Win32 structures
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

        # Flush stale events left over from previous user input
        kernel32.FlushConsoleInputBuffer(handle)

        buf = INPUT_RECORD()
        n = wintypes.DWORD(0)

        while self._running and not self._stop_event.is_set():
            if not kernel32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(n)) or n.value == 0:
                self._stop_event.wait(0.02)
                continue

            # Read one event from the buffer
            if not kernel32.ReadConsoleInputW(handle, ctypes.byref(buf), 1, ctypes.byref(n)) or n.value == 0:
                self._stop_event.wait(0.02)
                continue

            # Log ALL event types for diagnosis
            if buf.EventType != KEY_EVENT:
                _log.debug("Console event: type=%d (non-key, discarded)", buf.EventType)
                continue  # Discard mouse, resize, focus, menu events
            ke = buf.Event.KeyEvent
            if not ke.bKeyDown:
                continue  # Discard key-up events

            vk = ke.wVirtualKeyCode
            scan = ke.wVirtualScanCode
            ctrl = bool(ke.dwControlKeyState & CTRL_MASK)
            char_val = ke.uChar.UnicodeChar

            # Log every key-down event so we can diagnose phantom interrupts
            _log.info(
                "KEY_EVENT: vk=0x%02X scan=0x%02X ctrl=%s char=%r state=0x%08X",
                vk, scan, ctrl, char_val, ke.dwControlKeyState,
            )

            # Real Escape key: VK_ESCAPE with a non-zero scan code (0x01).
            # VT sequence bytes injected by the terminal (focus events,
            # arrow keys in VT mode, etc.) have scan code 0.
            if vk == VK_ESCAPE and scan != 0:
                _interrupt_state.trigger("escape")
            elif ctrl and vk == 0x42:  # Ctrl+B
                _interrupt_state.trigger_background()
            elif ctrl and vk == 0x43:  # Ctrl+C
                _interrupt_state.trigger("ctrl-c")
            # All other events are silently consumed and discarded

    def _monitor_windows_msvcrt(self):
        """Fallback Windows monitor using msvcrt (no Escape support)."""
        import msvcrt

        while self._running and not self._stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Skip Escape entirely in fallback — too unreliable via msvcrt
                if key == b'\x02':  # Ctrl+B
                    _interrupt_state.trigger_background()
                elif key == b'\x03':  # Ctrl+C
                    _interrupt_state.trigger("ctrl-c")
            self._stop_event.wait(0.02)
    
    def _monitor_unix(self):
        """Unix-specific keyboard monitoring."""
        import select
        import termios
        import tty
        
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
            while self._running and not self._stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x1b':  # Escape
                        _interrupt_state.trigger("escape")
                        # Don't break - keep monitoring
                    elif key == '\x02':  # Ctrl+B
                        _interrupt_state.trigger_background()
                    elif key == '\x03':  # Ctrl+C
                        _interrupt_state.trigger("ctrl-c")
        except:
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
    """Start keyboard monitoring."""
    get_monitor().start()


def stop_monitoring():
    """Stop keyboard monitoring."""
    get_monitor().stop()
