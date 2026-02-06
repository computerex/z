"""Keyboard interrupt handling for streaming operations."""

import sys
import threading
import asyncio
from typing import Optional
from dataclasses import dataclass


@dataclass
class InterruptState:
    """Shared state for interrupt handling."""
    interrupted: bool = False
    background: bool = False
    reason: str = ""
    
    def reset(self):
        self.interrupted = False
        self.background = False
        self.reason = ""
    
    def trigger(self, reason: str = "user"):
        self.interrupted = True
        self.reason = reason
    
    def trigger_background(self):
        self.background = True
        self.reason = "background"


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


class KeyboardMonitor:
    """Monitor for escape key and Ctrl+B during streaming."""
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start monitoring for keys."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        reset_interrupt()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.1)
            self._thread = None
    
    def _monitor_loop(self):
        """Monitor for keys in background thread."""
        if sys.platform == 'win32':
            self._monitor_windows()
        else:
            self._monitor_unix()
    
    def _monitor_windows(self):
        """Windows-specific keyboard monitoring."""
        import msvcrt
        
        while self._running and not self._stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Escape key = 0x1b (27)
                if key == b'\x1b':
                    _interrupt_state.trigger("escape")
                    break
                # Ctrl+B = 0x02
                elif key == b'\x02':
                    _interrupt_state.trigger_background()
                    break
            self._stop_event.wait(0.05)  # 50ms polling
    
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
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x1b':  # Escape
                        _interrupt_state.trigger("escape")
                        break
                    elif key == '\x02':  # Ctrl+B
                        _interrupt_state.trigger_background()
                        break
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
