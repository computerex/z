from harness.streaming_client import StreamingJSONClient, StreamingMessage


def test_provider_kind_detection():
    assert StreamingJSONClient("k", "https://api.anthropic.com/v1", "claude-3-5-sonnet")._provider_kind() == "anthropic"
    assert StreamingJSONClient("k", "https://openrouter.ai/api/v1", "openai/gpt-4o")._provider_kind() == "openrouter"
    assert StreamingJSONClient("k", "https://api.openai.com/v1", "gpt-4o")._provider_kind() == "openai"
    assert StreamingJSONClient("k", "https://api.z.ai/api/paas/v4", "glm-4.7")._provider_kind() == "openai_compat"


def test_build_anthropic_messages_adds_cache_breakpoints():
    client = StreamingJSONClient("k", "https://api.anthropic.com/v1", "claude-3-5-sonnet")
    messages = [
        StreamingMessage(role="system", content="system instructions"),
        StreamingMessage(role="user", content="u1"),
        StreamingMessage(role="assistant", content="a1"),
        StreamingMessage(role="user", content="u2"),
        StreamingMessage(role="assistant", content="a2"),
    ]

    system_blocks, anth_messages = client._build_anthropic_messages(messages, enable_prompt_caching=True)

    assert system_blocks
    assert system_blocks[-1]["cache_control"]["type"] == "ephemeral"
    assert len(anth_messages) == 4
    # Prefix cache breakpoint should be set before the latest two turns.
    assert anth_messages[1]["content"][-1]["cache_control"]["type"] == "ephemeral"
    # Latest assistant turn should not be marked cacheable by default.
    assert "cache_control" not in anth_messages[-1]["content"][-1]


def test_build_anthropic_messages_preserves_assistant_provider_blocks():
    client = StreamingJSONClient("k", "https://api.anthropic.com/v1", "claude-3-5-sonnet")
    messages = [
        StreamingMessage(role="system", content="sys"),
        StreamingMessage(role="user", content="question"),
        StreamingMessage(
            role="assistant",
            content="<thinking>t</thinking>\nanswer",
            provider_blocks=[
                {"type": "thinking", "thinking": "secret", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        ),
    ]
    system_blocks, anth_messages = client._build_anthropic_messages(messages, enable_prompt_caching=True)
    assert anth_messages[1]["role"] == "assistant"
    assert anth_messages[1]["content"][0]["type"] == "thinking"
    assert anth_messages[1]["content"][0]["signature"] == "sig"
    # Cache control should be applied to cacheable blocks, not thinking blocks.
    assert "cache_control" not in anth_messages[1]["content"][0]


def test_build_anthropic_messages_does_not_cache_mark_empty_text_block():
    client = StreamingJSONClient("k", "https://api.anthropic.com/v1", "claude-3-5-sonnet")
    messages = [
        StreamingMessage(role="system", content="sys"),
        StreamingMessage(role="user", content="u1"),
        StreamingMessage(
            role="assistant",
            content="placeholder",
            provider_blocks=[
                {"type": "thinking", "thinking": "hidden", "signature": "sig"},
                {"type": "text", "text": "   "},
            ],
        ),
    ]

    _system_blocks, anth_messages = client._build_anthropic_messages(messages, enable_prompt_caching=True)
    assistant_blocks = anth_messages[-1]["content"]
    assert assistant_blocks[0]["type"] == "thinking"
    assert "cache_control" not in assistant_blocks[0]
    assert assistant_blocks[1]["type"] == "text"
    assert "cache_control" not in assistant_blocks[1]


def test_normalize_openai_usage_preserves_reasoning_and_cache_details():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 60},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }
    out = StreamingJSONClient._normalize_openai_usage(usage)
    assert out["prompt_tokens"] == 100
    assert out["completion_tokens"] == 20
    assert out["prompt_cached_tokens"] == 60
    assert out["completion_reasoning_tokens"] == 7


def test_minimax_reasoning_details_extraction_flattens_text_blocks():
    client = StreamingJSONClient("k", "https://api.minimax.io/v1", "MiniMax-M2.5")
    assert client._is_minimax_provider() is True
    text = client._extract_reasoning_details_text([
        {"type": "reasoning.summary", "text": "plan "},
        {"type": "reasoning.text", "text": "step 1"},
        {"type": "reasoning.text", "ignored": "x"},
        "skip",
    ])
    assert text == "plan step 1"


def test_minimax_reasoning_details_extraction_handles_nested_message_shape():
    text = StreamingJSONClient._extract_reasoning_details_text({
        "role": "assistant",
        "reasoning_details": [
            {"type": "reasoning.summary", "text": "Thinking: "},
            {"type": "reasoning.text", "text": "hello"},
        ],
    })
    assert text == "Thinking: hello"
