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

    _SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    _ACCENT = "\033[38;5;75m"

    def __init__(self, enabled: bool = True, safe_mode: bool = None):
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
        # Turn-level timer (persists across clear() calls within a turn)
        self._turn_start: Optional[float] = None
        # Retry info
        self._retry_attempt = 0
        self._retry_max = 0
        self._retry_wait = 0
        # Background ticker for spinner animation and elapsed time
        self._tick_stop = threading.Event()
        self._ticker_thread: Optional[threading.Thread] = None
        # Safe mode indicator (auto-detect from env if not specified)
        import os
        self._safe_mode = safe_mode if safe_mode is not None else os.environ.get("HARNESS_SAFE_MODE") == "1"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_iterations(self, current: int, maximum: int):
        """Set current iteration counter for display."""
        self._iteration = current
        self._max_iterations = maximum

    def set_turn_start(self):
        """Mark the start of a new agent turn (total runtime timer)."""
        self._turn_start = time.time()

    def clear_turn(self):
        """Clear the turn timer (call when the turn is fully done)."""
        self._turn_start = None

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

        # Turn-level elapsed time
        turn_elapsed_str = ""
        if self._turn_start:
            turn_elapsed_str = self._format_elapsed(now - self._turn_start)

        # Build compact line: [SAFE] icon text │ iter │ phase_elapsed │ total_elapsed
        safe_prefix = "\033[93m[SAFE]\033[0m\033[2m " if self._safe_mode else ""
        parts = [f"{safe_prefix}{icon} {self._text}"]
        if self._max_iterations > 0:
            parts.append(f"iter: {self._iteration}/{self._max_iterations}")
        if elapsed_str:
            parts.append(elapsed_str)
        if turn_elapsed_str and turn_elapsed_str != elapsed_str:
            parts.append(f"total {turn_elapsed_str}")
        
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
        """Get the current icon/spinner character with accent coloring."""
        c = self._ACCENT
        r = "\033[0m\033[2m"
        if self._state in (self.STREAMING,):
            return f"{c}\u25cf{r}"
        if self._state == self.IDLE:
            return f"{c}\u25cb{r}"
        if self._state == self.RETRYING:
            return f"{c}\u27f3{r}"
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

    def print_safe(self, console, *args, **kwargs):
        """Print to console while temporarily clearing the status line.
        
        Usage:
            status.print_safe(console, "Hello world")
            status.print_safe(console, Panel(...))
        """
        if not self._enabled or not self._visible:
            console.print(*args, **kwargs)
            return
        
        # Clear status line before printing
        with self._lock:
            if self._visible:
                cols = shutil.get_terminal_size().columns
                sys.stdout.write(f"\r{' ' * cols}\r")
                sys.stdout.flush()
                self._visible = False
        
        # Print the content
        console.print(*args, **kwargs)
        
        # Move to new line and restore status line at the bottom
        with self._lock:
            if self._text:  # Only if we have status to show
                sys.stdout.write("\n")
                self._render()

    def wrap_console(self, console):
        """Return a wrapped console that clears/restores status line on print.
        
        Usage:
            wrapped_console = status.wrap_console(console)
            wrapped_console.print("Hello")  # Status line handled automatically
        """
        if not self._enabled:
            return console
        
        class StatusAwareConsole:
            def __init__(status_self, status_line, real_console):
                status_self._status = status_line
                status_self._console = real_console
            
            def print(status_self, *args, **kwargs):
                status_self._status.print_safe(status_self._console, *args, **kwargs)
            
            def __getattr__(status_self, name):
                # Delegate all other attributes to the real console
                return getattr(status_self._console, name)
        
        return StatusAwareConsole(self, console)
