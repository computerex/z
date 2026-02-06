"""Keyboard interrupt handling for streaming operations."""

import sys
import threading
import signal
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
            return
        
        self._running = True
        self._stop_event.clear()
        reset_interrupt()
        
        # Install Ctrl+C handler
        self._original_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def _sigint_handler(self, signum, frame):
        """Handle Ctrl+C by triggering interrupt."""
        _interrupt_state.trigger("ctrl-c")
        # Re-raise on second Ctrl+C for hard exit
        if _interrupt_state.interrupted:
            # Already interrupted once, this is the second time
            if self._original_sigint and self._original_sigint != signal.SIG_DFL:
                self._original_sigint(signum, frame)
            else:
                raise KeyboardInterrupt()
    
    def stop(self):
        """Stop monitoring."""
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
        """Windows-specific keyboard monitoring."""
        import msvcrt
        
        while self._running and not self._stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # Escape key = 0x1b (27)
                if key == b'\x1b':
                    _interrupt_state.trigger("escape")
                    # Don't break - keep monitoring for more keys
                # Ctrl+B = 0x02
                elif key == b'\x02':
                    _interrupt_state.trigger_background()
                # Ctrl+C = 0x03
                elif key == b'\x03':
                    _interrupt_state.trigger("ctrl-c")
            self._stop_event.wait(0.02)  # 20ms polling for more responsiveness
    
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
