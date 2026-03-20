"""Tests for provider setup and model fetching."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
import json


# Import from harness.py directly (it's at root level, not in src/harness/)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "harness",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness.py"
    ),
)
harness_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness_module)


class TestFetchModelsFromProviderApi:
    """Test the _fetch_models_from_provider_api function."""

    def test_successful_fetch(self):
        """Test successful model fetch from provider API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "claude-3-opus", "object": "model"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = harness_module._fetch_models_from_provider_api(
                "https://api.example.com/v1", "sk-test-key"
            )

        assert result == ["claude-3-opus", "gpt-3.5-turbo", "gpt-4"]
        mock_response.raise_for_status.assert_called_once()

    def test_empty_response(self):
        """Test handling of empty model list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = harness_module._fetch_models_from_provider_api(
                "https://api.example.com/v1", "sk-test-key"
            )

        assert result == []

    def test_api_error_fallback(self):
        """Test that API errors return empty list for manual entry."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = harness_module._fetch_models_from_provider_api(
                "https://api.example.com/v1", "sk-test-key"
            )

        assert result == []

    def test_no_api_key(self):
        """Test fetch without API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "model-1"}, {"id": "model-2"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = harness_module._fetch_models_from_provider_api(
                "https://api.example.com/v1",
                "",  # No API key
            )

        # Verify no Authorization header was sent
        call_kwargs = mock_get.call_args[1]
        assert "Authorization" not in call_kwargs.get("headers", {})

    def test_oauth_token_not_sent(self):
        """Test that oauth: tokens are not sent as Bearer tokens."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "model-1"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = harness_module._fetch_models_from_provider_api(
                "https://api.example.com/v1",
                "oauth:some-token",  # OAuth token
            )

        # Verify no Authorization header was sent for OAuth tokens
        call_kwargs = mock_get.call_args[1]
        assert "Authorization" not in call_kwargs.get("headers", {})


class TestChooseProviderPresetInteractive:
    """Test the _choose_provider_preset_interactive function."""

    @patch("builtins.input")
    @patch("rich.console.Console.print")
    def test_custom_provider_with_profile_name(self, mock_print, mock_input):
        """Test that custom provider accepts profile name."""
        # Simulate: choice 25, URL input, profile name input
        mock_input.side_effect = ["25", "https://nano-gpt.com/api/v1", "nano-gpt"]

        result = harness_module._choose_provider_preset_interactive("", "")

        preset_key, label, api_url, default_model, profile_name = result
        assert preset_key == "custom"
        assert label == "Custom"
        assert api_url == "https://nano-gpt.com/api/v1/"
        assert profile_name == "nano-gpt"

    @patch("builtins.input")
    @patch("rich.console.Console.print")
    def test_custom_provider_default_profile_name(self, mock_print, mock_input):
        """Test that custom provider defaults to 'default' if no name given."""
        mock_input.side_effect = [
            "25",
            "https://api.example.com/v1",
            "",
        ]  # Empty profile name

        result = harness_module._choose_provider_preset_interactive("", "")

        _, _, _, _, profile_name = result
        assert profile_name == "default"

    @patch("builtins.input")
    @patch("rich.console.Console.print")
    def test_preset_provider_returns_empty_profile_name(self, mock_print, mock_input):
        """Test that preset providers return empty profile name."""
        mock_input.side_effect = ["8"]  # Select OpenAI (option 8)

        result = harness_module._choose_provider_preset_interactive("", "")

        _, _, _, _, profile_name = result
        assert profile_name == ""


class TestFetchProviderModelIds:
    """Test the _fetch_provider_model_ids function routes correctly."""

    @patch.object(harness_module, "_fetch_models_from_provider_api")
    @patch("harness.streaming_client.search_litellm_models")
    def test_known_provider_uses_litellm(self, mock_search, mock_fetch_provider):
        """Test that known providers use LiteLLM model search."""
        harness_module._fetch_provider_model_ids("https://api.anthropic.com", "sk-test")

        # Should call LiteLLM search for known providers
        mock_search.assert_called_once_with("anthropic/")
        mock_fetch_provider.assert_not_called()

    @patch.object(harness_module, "_fetch_models_from_provider_api")
    @patch("harness.streaming_client.search_litellm_models")
    def test_unknown_provider_queries_api(self, mock_search, mock_fetch_provider):
        """Test that unknown providers query their API directly."""
        mock_fetch_provider.return_value = ["model-1", "model-2"]

        result = harness_module._fetch_provider_model_ids(
            "https://custom-api.example.com/v1", "sk-test"
        )

        # Should query the provider's API for unknown/custom URLs
        mock_fetch_provider.assert_called_once_with(
            "https://custom-api.example.com/v1", "sk-test"
        )
        mock_search.assert_not_called()
        assert result == ["model-1", "model-2"]
