"""Tests for replaying buffered sub-agent output when focusing an agent."""

import io

from harness.sub_agent_manager import TeeWriter


def test_replay_to_real_replays_background_output():
    real = io.StringIO()
    tee = TeeWriter(real)
    tee.write("background output\n")

    assert tee.replay_to_real() == len("background output\n")
    assert real.getvalue() == "background output\n"


def test_replay_to_real_truncates_old_output():
    real = io.StringIO()
    tee = TeeWriter(real)
    tee.write("x" * 100)

    tee.replay_to_real(max_chars=10)

    assert real.getvalue() == "[... earlier output omitted ...]\n" + "x" * 10 + "\n"
