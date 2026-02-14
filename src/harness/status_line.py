"""Persistent bottom status line for the harness terminal UI.

Renders a single overwritable line at the bottom of the terminal showing
the current activity state: waiting for LLM, streaming, executing tool,
retrying, idle, etc.  Uses simple \\r + ANSI clear-line to avoid complex
terminal manipulation (no curses / rich Live needed).
"""

import sys
import time
import shutil
import threading
from typing import Optional


class StatusLine:
    """A single persistent status line at the bottom of the terminal.
    
    Usage:
        status = StatusLine()
        status.update("Sending request to LLM...")
        # ... LLM starts streaming ...
        status.update("Streaming response", streaming=True)
        # ... done ...
        status.clear()
    """

    # State constants
    IDLE = "idle"
    SENDING = "sending"
    STREAMING = "streaming"
    TOOL_EXEC = "tool_exec"
    RETRYING = "retrying"
    COMPACTING = "compacting"
    WAITING = "waiting"

    # Spinner frames for active states (minimal dots)
    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    # Cyan ANSI color for spinner/icon
    _CYAN = "\033[36m"

    def __init__(self, enabled: bool = True):
        self._enabled = enabled and sys.stdout.isatty()
        self._text = ""
        self._state = self.IDLE
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._spinner_idx = 0
        self._last_render = 0.0
        self._visible = False
        # Track active iteration for display
        self._iteration = 0
        self._max_iterations = 0
        # Retry info
        self._retry_attempt = 0
        self._retry_max = 0
        self._retry_wait = 0
        # Background ticker for spinner animation and elapsed time
        self._tick_stop = threading.Event()
        self._ticker_thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_iterations(self, current: int, maximum: int):
        """Set current iteration counter for display."""
        self._iteration = current
        self._max_iterations = maximum

    def update(self, text: str, state: str = WAITING):
        """Update the status line text and state."""
        if not self._enabled:
            return
        with self._lock:
            self._text = text
            self._state = state
            if self._start_time is None:
                self._start_time = time.time()
            self._render()
        self._start_ticker()

    def set_retry(self, attempt: int, max_retries: int, wait_secs: float, reason: str = ""):
        """Show retry status."""
        if not self._enabled:
            return
        with self._lock:
            self._retry_attempt = attempt
            self._retry_max = max_retries
            self._retry_wait = wait_secs
            reason_part = f" ({reason})" if reason else ""
            self._text = f"Retry {attempt}/{max_retries}{reason_part} — waiting {wait_secs:.0f}s"
            self._state = self.RETRYING
            self._render()
        self._start_ticker()

    def clear(self):
        """Clear the status line and reset state."""
        if not self._enabled:
            return
        self._stop_ticker()
        with self._lock:
            if self._visible:
                cols = shutil.get_terminal_size().columns
                sys.stdout.write(f"\r{' ' * cols}\r")
                sys.stdout.flush()
                self._visible = False
            self._text = ""
            self._state = self.IDLE
            self._start_time = None

    def _render(self):
        """Render the status line (must be called under lock)."""
        now = time.time()
        # Throttle renders to ~10fps
        if now - self._last_render < 0.1 and self._visible:
            return
        self._last_render = now

        elapsed = now - self._start_time if self._start_time else 0
        elapsed_str = self._format_elapsed(elapsed)

        # Pick icon based on state
        icon = self._get_icon()

        # Build compact line: icon text │ iter │ elapsed
        parts = [f"{icon} {self._text}"]
        if self._max_iterations > 0:
            parts.append(f"iter: {self._iteration}/{self._max_iterations}")
        if elapsed_str:
            parts.append(elapsed_str)
        
        cols = shutil.get_terminal_size().columns
        line = " " + " │ ".join(parts) + " "

        # Truncate if too wide
        if len(line) > cols:
            line = line[:cols - 1] + "…"

        # Dim + cyan icon via ANSI
        dim = "\033[2m"
        reset = "\033[0m"

        sys.stdout.write(f"\r{dim}{line:<{cols}}{reset}\r")
        sys.stdout.flush()
        self._visible = True

    def _get_icon(self) -> str:
        """Get the current icon/spinner character with cyan coloring."""
        c = self._CYAN
        r = "\033[0m\033[2m"  # reset to dim (we're inside a dim block)
        if self._state in (self.STREAMING,):
            return f"{c}●{r}"
        if self._state == self.IDLE:
            return f"{c}○{r}"
        if self._state == self.RETRYING:
            return f"{c}⟳{r}"
        # Animated spinner for all other active states
        self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER)
        return f"{c}{self._SPINNER[self._spinner_idx]}{r}"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time compactly."""
        if seconds < 1:
            return ""
        if seconds < 60:
            return f"{seconds:.0f}s"
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins}m{secs:02d}s"

    def tick(self):
        """Re-render the status line (call periodically to animate spinner/timer)."""
        if not self._enabled or not self._text:
            return
        with self._lock:
            self._render()

    def _start_ticker(self):
        """Start the background ticker for spinner/timer animation."""
        if self._ticker_thread and self._ticker_thread.is_alive():
            return
        self._tick_stop.clear()
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()

    def _stop_ticker(self):
        """Stop the background ticker."""
        self._tick_stop.set()
        if self._ticker_thread:
            self._ticker_thread.join(timeout=0.5)
            self._ticker_thread = None

    def _ticker_loop(self):
        """Background thread that animates the spinner and elapsed time."""
        while not self._tick_stop.wait(0.12):
            if self._text:
                self.tick()
