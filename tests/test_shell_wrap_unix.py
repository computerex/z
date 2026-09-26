"""Regression tests: Unix shell wrapper must redirect ALL of a compound
command's output to the log file.

The old wrapper was `f'{command} > "{log}" 2>&1'` — no grouping. For
`A; B && C` POSIX sh binds the redirect to the last segment only
(`A; B && C > log`), so earlier segments wrote to DEVNULL and their output
never reached the live display OR the tool result. Symptom: only the last
command's output "showed" for compound commands while single commands
worked fine.
"""

import os
import platform
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from harness.tools.shell import _wrap_shell_command


def test_unix_wrapper_groups_compound_commands(monkeypatch):
    """The whole command must sit inside a group before the redirect."""
    import harness.tools.shell as sh

    monkeypatch.setattr(sh.platform, "system", lambda: "Linux")

    wrapped = _wrap_shell_command(
        object(), "git rev-parse HEAD && git log --oneline -1; echo done", "/tmp/z.log"
    )
    # Redirect must apply to a GROUP containing the entire command, not
    # just the final segment.
    assert wrapped.startswith("(\n"), f"missing subshell group: {wrapped!r}"
    assert wrapped.endswith('\n) > "/tmp/z.log" 2>&1'), f"bad redirect: {wrapped!r}"
    # The compound command itself must be intact between the group delimiters
    body = wrapped[len("(\n") : wrapped.rindex("\n) > ")]
    assert body == "git rev-parse HEAD && git log --oneline -1; echo done"


def test_unix_wrapper_survives_trailing_comment(monkeypatch):
    """A trailing `# comment` must not swallow the closing paren."""
    import harness.tools.shell as sh

    monkeypatch.setattr(sh.platform, "system", lambda: "Darwin")

    wrapped = _wrap_shell_command(object(), "echo hi # note", "/tmp/z.log")
    # The `)` must be on its own line AFTER the newline, not inside the comment
    assert wrapped == '(\necho hi # note\n) > "/tmp/z.log" 2>&1'


def test_windows_routes_unchanged(monkeypatch):
    """Windows cmd/PowerShell routes already group correctly (cmd /c quotes,
    PowerShell launcher is a single process) — ensure they are untouched."""
    import harness.tools.shell as sh

    monkeypatch.setattr(sh.platform, "system", lambda: "Windows")
    monkeypatch.delenv("HARNESS_WINDOWS_SHELL", raising=False)

    # PowerShell route (default): EncodedCommand launcher, process-level redirect
    wrapped = _wrap_shell_command(object(), "echo hello", "C:/tmp/z.log")
    assert "powershell" in wrapped and "-EncodedCommand" in wrapped
    assert wrapped.endswith('> "C:/tmp/z.log" 2>&1')

    # cmd.exe route: quotes group the compound command
    monkeypatch.setenv("HARNESS_WINDOWS_SHELL", "cmd")
    wrapped = _wrap_shell_command(object(), "echo a && echo b", "C:/tmp/z.log")
    assert wrapped == 'cmd /c "echo a && echo b" > "C:/tmp/z.log" 2>&1'


@pytest.mark.skipif(platform.system() == "Windows", reason="needs POSIX sh")
def test_compound_output_actually_reaches_log():
    """End-to-end through a real /bin/sh: every segment's output lands in
    the log file. (Runs on Linux/macOS; the equivalent manual proof on this
    repo's CI host is via WSL.)"""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tf:
        log = tf.name
    try:
        wrapped = _wrap_shell_command(object(), "echo FIRST; echo SECOND", log)
        # Wrapped commands are built for create_subprocess_shell (POSIX sh)
        subprocess.run(wrapped, shell=True, check=True)
        with open(log, "r") as f:
            contents = f.read()
        assert "FIRST" in contents and "SECOND" in contents
    finally:
        os.unlink(log)
