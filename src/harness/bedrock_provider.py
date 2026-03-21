"""Amazon Bedrock provider integration for harness.

Adds support for AWS Bedrock models including Qwen, Claude, Nova, etc.
Uses boto3 with bearer token authentication.
"""

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

    def is_available(self) -> bool:
        """Check if Bedrock client can be initialized."""
        if not BOTO3_AVAILABLE:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def chat(
        self,
        messages: List[BedrockMessage],
        on_content: Optional[callable] = None,
        on_thinking: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Send chat request to Bedrock.

        Returns dict with:
        - content: str
        - thinking: Optional[str]
        - usage: Dict[str, int]
        - finish_reason: str
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

        # Call Bedrock, auto-retry with us. prefix if bare model ID is rejected
        try:
            response = client.converse(**request)
        except client.exceptions.ValidationException as e:
            if "inference profile" in str(e).lower() and not model_id.startswith(("us.", "eu.", "ap.", "global.")):
                model_id = f"us.{model_id}"
                request["modelId"] = model_id
                self.model = model_id  # cache for future calls
                response = client.converse(**request)
            else:
                raise

        # Parse response
        result = {
            "content": "",
            "thinking": None,
            "usage": {},
            "finish_reason": "stop",
        }

        # Extract content
        if "output" in response and "message" in response["output"]:
            message = response["output"]["message"]
            content_blocks = message.get("content", [])

            text_parts = []
            thinking_parts = []

            for block in content_blocks:
                if isinstance(block, dict):
                    if "text" in block:
                        text_parts.append(block["text"])
                        if on_content:
                            on_content(block["text"])
                    elif "reasoningContent" in block:
                        # Handle reasoning content if present
                        reasoning = block["reasoningContent"]
                        if isinstance(reasoning, dict) and "text" in reasoning:
                            thinking_parts.append(reasoning["text"])
                            if on_thinking:
                                on_thinking(reasoning["text"])

            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["thinking"] = "".join(thinking_parts)

        # Extract usage
        if "usage" in response:
            usage = response["usage"]
            result["usage"] = {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }

        # Extract stop reason
        if "stopReason" in response:
            result["finish_reason"] = response["stopReason"].lower()

        return result

    def chat_stream(
        self,
        messages: List[BedrockMessage],
        on_content: Optional[callable] = None,
        on_thinking: Optional[callable] = None,
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

        return {
            "content": "".join(content_buffer),
            "thinking": "".join(thinking_buffer) if thinking_buffer else None,
            "usage": usage,
            "finish_reason": finish_reason,
        }


def is_bedrock_url(url: str) -> bool:
    """Check if URL is a Bedrock endpoint."""
    return "bedrock" in url.lower() and "amazonaws.com" in url.lower()


def is_bedrock_model(model: str) -> bool:
    """Check if model ID is a Bedrock model."""
    # Strip region prefix (us., eu., ap., global.) before checking
    m = model.lower()
    for prefix in ("us.", "eu.", "ap.", "global."):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    bedrock_prefixes = (
        "qwen.",
        "anthropic.",
        "amazon.",
        "meta.",
        "cohere.",
        "ai21.",
        "stability.",
        "mistral.",
        "moonshot.",
    )
    return any(m.startswith(p) for p in bedrock_prefixes)


# Convenience function to create client from URL
def create_bedrock_client(
    base_url: str, api_key: str, model: str, **kwargs
) -> Optional[BedrockClient]:
    """Create Bedrock client if URL/model indicates Bedrock."""
    if is_bedrock_url(base_url) or is_bedrock_model(model):
        # Extract region from URL if present
        region = "us-east-1"
        if ".amazonaws.com" in base_url:
            parts = base_url.split(".")
            if len(parts) >= 2:
                potential_region = parts[1]
                if potential_region.startswith("us-") or potential_region.startswith(
                    "eu-"
                ):
                    region = potential_region

        return BedrockClient(api_key=api_key, region=region, model=model, **kwargs)
    return None


def list_bedrock_models(api_key: str, region: str = "us-east-1") -> List[str]:
    """List available foundation models from AWS Bedrock.

    Queries the actual AWS Bedrock API to get all models available
    in the user's account/region.

    Args:
        api_key: The bearer token for authentication
        region: AWS region (default: us-east-1)

    Returns:
        List of model IDs available on Bedrock
    """
    if not BOTO3_AVAILABLE:
        return []

    try:
        # Set up bearer token
        if api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

        # Create bedrock client (not runtime - we need the management API)
        client = boto3.client(
            "bedrock",
            region_name=region,
            config=Config(retries={"max_attempts": 3}),
        )

        # List foundation models (direct call, paginator not supported)
        models = []
        response = client.list_foundation_models()

        for model in response.get("modelSummaries", []):
            model_id = model.get("modelId", "")
            if model_id:
                models.append(model_id)

        # Also include inference profile IDs (required for newer models)
        try:
            profile_resp = client.list_inference_profiles(maxResults=1000)
            for p in profile_resp.get("inferenceProfileSummaries", []):
                pid = p.get("inferenceProfileId", "")
                if pid and p.get("status") == "ACTIVE":
                    models.append(pid)
        except Exception:
            pass  # Inference profiles API may not be available

        return sorted(set(models))
    except Exception as e:
        # Debug: print error to stderr so we can see what's failing
        import sys

        print(f"[DEBUG] Bedrock model list error: {e}", file=sys.stderr)
        # Fallback: return empty list if we can't fetch
        return []
