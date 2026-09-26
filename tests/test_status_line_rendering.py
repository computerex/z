"""Regression test: prompt-scoped patch_stdout must not wrap status-line writes.

The earlier always-on patch_stdout made every StatusLine spinner frame render
as its own line with escape bytes eaten. This test pins the StatusLine contract:
renders must go to the *current* sys.stdout with raw ANSI and a leading \r.
"""

import io
import sys

from harness.status_line import StatusLine


def test_status_line_writes_raw_ansi_with_carriage_return(monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    status = StatusLine(enabled=True)
    status._safe_mode = False
    status.set_iterations(1, 500)
    status._start_time = 1.0
    status._text = "Sending to LLM"
    status._state = StatusLine.SENDING
    status._visible = False
    status._last_render = 0.0

    status._render()

    out = captured.getvalue()
    assert out.startswith("\r"), "spinner frames must overwrite in place"
    assert "\x1b[2m" in out, "dim ANSI must be preserved raw"
    assert "iter: 1/500" in out
    assert "\n" not in out, "status renders must not emit newlines"


def test_status_line_disabled_in_non_tty(monkeypatch):
    class _NotATty(io.StringIO):
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdout", _NotATty())
    status = StatusLine(enabled=True)
    assert status.enabled is False
