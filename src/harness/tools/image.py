"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from urllib.parse import urlparse
from ..image_utils import encode_image_to_data_uri, is_image_file

async def analyze_image(self, params: Dict[str, str]) -> str:
    """Analyze an image using the Z.AI GLM-4.6V vision model.

    This tool is only available when using the glm-4.7 model on Z.AI.
    """
    import httpx
    from urllib.parse import urlparse
    from ..image_utils import encode_image_to_data_uri, is_image_file

    path_str = params.get("path", "")
    question = params.get(
        "question",
        "Describe this image in detail. Note any text, UI elements, errors, or important visual details.",
    )

    from .mcp import _resolve_path
    path = _resolve_path(self, path_str)
    if not path.exists():
        return f"Error: Image not found: {path}"
    if not is_image_file(path):
        return f"Error: Unsupported image format: {path.suffix}. Use jpg, jpeg, png, gif, or webp."

    try:
        data_uri = encode_image_to_data_uri(path)
        parsed = urlparse(self.config.api_url)
        vision_url = f"{parsed.scheme}://{parsed.netloc}/api/coding/paas/v4/chat/completions"

        payload = {
            "model": "glm-4.6v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(vision_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
                    ctx_id = self.context.add("image_analysis", str(path), content)
                    return f"[Context ID: {ctx_id}]\n\nImage: {path_str}\n\n{content}"
                return "Vision model returned empty response."
            return f"Unexpected response format: {data}"
    except httpx.HTTPStatusError as e:
        return f"Error calling vision API: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        return f"Error analyzing image: {e}"

