"""Prompt session completion uses the live sub-agent name provider."""

from pathlib import Path

from harness.main import create_prompt_session
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput


def test_prompt_session_completes_live_agent_names():
    names = ["alpha"]
    workspace = Path(__file__).resolve().parent
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        session = create_prompt_session(
            workspace / ".history",
            workspace,
            agent_names_fn=lambda: names,
        )

    assert session is not None
    document = Document("/agent a")
    event = CompleteEvent()
    assert [item.text for item in session.completer.get_completions(document, event)] == [
        "alpha"
    ]

    names.append("amber")
    assert [item.text for item in session.completer.get_completions(document, event)] == [
        "alpha",
        "amber",
    ]
