"""Amazon Bedrock streaming client (fallback for non-LiteLLM Bedrock access)."""

import os
import json
import base64 as _base64
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Try to import boto3, but don't fail if not installed
try:
    import boto3
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


@dataclass
class BedrockMessage:
    """Message format for Bedrock converse API."""

    role: str
    content: Union[str, List[Dict[str, Any]]]

    def to_bedrock_format(self) -> Dict[str, Any]:
        """Convert to Bedrock's message format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": [{"text": self.content}]}
        else:
            # Convert OpenAI format to Bedrock format
            bedrock_content = []
            for part in self.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        bedrock_content.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        # Handle images if needed
                        image_url = part.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                            if url.startswith("data:"):
                                # Extract base64 image
                                header, b64 = url.split(",", 1)
                                media_type = header[5:].split(";", 1)[0] or "image/png"
                                bedrock_content.append(
                                    {
                                        "image": {
                                            "format": media_type.split("/")[-1],
                                            "source": {"bytes": _base64.b64decode(b64)},
                                        }
                                    }
                                )
            return {
                "role": self.role,
                "content": bedrock_content
                if bedrock_content
                else [{"text": str(self.content)}],
            }


class BedrockClient:
    """Client for AWS Bedrock Runtime API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = "us-east-1",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.region = region
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        # Set up bearer token
        if api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        elif "BED_ROCK_API_KEY" in os.environ:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["BED_ROCK_API_KEY"]

    def _get_client(self):
        """Get or create boto3 client."""
        if self._client is None:
            if not BOTO3_AVAILABLE:
                raise ImportError(
                    "boto3 is required for Bedrock. Install with: pip install boto3"
                )

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(retries={"max_attempts": 3}),
            )
        return self._client

    def _resolve_model_id(self) -> str:
        """Resolve model ID, adding region prefix for inference profiles if needed.

        Newer Bedrock models require inference profile IDs (us.model or global.model)
        instead of bare model IDs. If the model already has a region prefix, return as-is.
        """
        model = self.model
        # Already has a region prefix
        if model.startswith(("us.", "eu.", "ap.", "global.")):
            return model
        return model

    def chat_stream(
        self,
        messages: List[BedrockMessage],
        on_content: Optional[callable] = None,
        on_thinking: Optional[callable] = None,
        tools: Optional[list] = None,
    ):
        """Stream chat response from Bedrock.

        Note: Bedrock's converse_stream API streams content.
        """
        client = self._get_client()

        # Convert messages to Bedrock format
        bedrock_messages = [
            m.to_bedrock_format() for m in messages if m.role != "system"
        ]

        # Extract system message if present
        system_content = None
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_content = [{"text": m.content}]
                break

        # Build request
        model_id = self._resolve_model_id()
        request = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
            },
        }

        if system_content:
            request["system"] = system_content

        # Convert OpenAI tool format to Bedrock toolConfig format
        if tools:
            bedrock_tools = []
            for t in tools:
                fn = t.get("function", {})
                bedrock_tools.append({
                    "toolSpec": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "inputSchema": {
                            "json": fn.get("parameters", {}),
                        },
                    }
                })
            if bedrock_tools:
                request["toolConfig"] = {"tools": bedrock_tools}

        # Stream from Bedrock, auto-retry with us. prefix if bare model ID is rejected
        try:
            response = client.converse_stream(**request)
        except client.exceptions.ValidationException as e:
            if "inference profile" in str(e).lower() and not model_id.startswith(("us.", "eu.", "ap.", "global.")):
                model_id = f"us.{model_id}"
                request["modelId"] = model_id
                self.model = model_id  # cache for future calls
                response = client.converse_stream(**request)
            else:
                raise

        content_buffer = []
        thinking_buffer = []
        usage = {}
        finish_reason = "stop"

        # Process stream
        stream = response.get("stream", [])
        _event_count = 0
        _reasoning_event_count = 0
        for event in stream:
            _event_count += 1
            # Handle content block delta
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    text = delta["text"]
                    content_buffer.append(text)
                    if on_content:
                        on_content(text)
                elif "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    if "text" in reasoning:
                        reasoning_text = reasoning["text"]
                        thinking_buffer.append(reasoning_text)
                        if on_thinking:
                            on_thinking(reasoning_text)

            # Handle metadata
            elif "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    u = metadata["usage"]
                    usage = {
                        "prompt_tokens": u.get("inputTokens", 0),
                        "completion_tokens": u.get("outputTokens", 0),
                        "total_tokens": u.get("totalTokens", 0),
                    }
                    # Propagate Bedrock cache-token fields so the cost
                    # tracker can apply discounted pricing for cache hits.
                    for _bk_key, _bk_api in (
                        ("cache_read_input_tokens", "cacheReadInputTokens"),
                        ("cache_creation_input_tokens", "cacheCreationInputTokens"),
                    ):
                        _bk_val = u.get(_bk_api)
                        if _bk_val is not None:
                            usage[_bk_key] = _bk_val

        return {
            "content": "".join(content_buffer),
            "thinking": "".join(thinking_buffer) if thinking_buffer else None,
            "usage": usage,
            "finish_reason": finish_reason,
        }


