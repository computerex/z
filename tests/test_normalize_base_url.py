"""Unit tests for _normalize_base_url (opencode-style full-endpoint base URLs)."""

import sys

sys.path.insert(0, "src")

from harness.streaming_client import _normalize_base_url


def test_strips_chat_completions_suffix():
    assert (
        _normalize_base_url("https://opencode.ai/inference/go/openai/v1/chat/completions")
        == "https://opencode.ai/inference/go/openai/v1"
    )


def test_strips_suffix_with_trailing_slash():
    assert (
        _normalize_base_url("https://opencode.ai/inference/go/openai/v1/chat/completions/")
        == "https://opencode.ai/inference/go/openai/v1"
    )


def test_normal_base_url_untouched():
    assert (
        _normalize_base_url("https://api.together.xyz/v1/")
        == "https://api.together.xyz/v1"
    )


def test_v1_base_untouched():
    assert _normalize_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"


def test_empty():
    assert _normalize_base_url("") == ""
    assert _normalize_base_url(None) == ""


def test_case_insensitive():
    assert (
        _normalize_base_url("https://x.example.COM/v1/Chat/Completions")
        == "https://x.example.COM/v1"
    )


def test_anthropic_messages_suffix():
    assert (
        _normalize_base_url("https://opencode.ai/inference/go/anthropic/v1/messages")
        == "https://opencode.ai/inference/go/anthropic/v1"
    )


def test_no_double_strip():
    # A base that only consists of the endpoint-ish tail stays a valid URL
    assert _normalize_base_url("https://host/chat/completions") == "https://host"
