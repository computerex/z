"""Regression tests for DeepSeek reasoning-content message round-tripping."""

from harness.streaming_client import StreamingJSONClient, StreamingMessage


def test_streaming_message_serializes_explicit_reasoning_content():
    message = StreamingMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
        reasoning_content="reasoning from the prior turn",
    )

    assert message.to_dict() == {
        "role": "assistant",
        "content": None,
        "tool_calls": message.tool_calls,
        "reasoning_content": "reasoning from the prior turn",
    }


def test_deepseek_assistant_messages_include_empty_reasoning_content():
    client = StreamingJSONClient(
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/deepseek-v4p1-flash",
    )
    message = StreamingMessage(role="assistant", content="prior response")

    serialized = message.to_dict()
    if "deepseek" in client.model.lower() and message.role == "assistant":
        serialized.setdefault("reasoning_content", "")

    assert serialized["reasoning_content"] == ""


def test_non_deepseek_assistant_message_does_not_gain_reasoning_content():
    client = StreamingJSONClient(
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/glm-5p3",
    )
    message = StreamingMessage(role="assistant", content="prior response")

    serialized = message.to_dict()
    if "deepseek" in client.model.lower() and message.role == "assistant":
        serialized.setdefault("reasoning_content", "")

    assert "reasoning_content" not in serialized
