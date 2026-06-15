"""Shared image utilities for vision-capable and non-vision models.

Provides a single code path for:
- Detecting image content blocks and image files
- Encoding images to data URIs
- Adapting user content so non-vision models never receive raw image blocks
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger("harness.image_utils")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def is_image_file(path: Union[str, Path]) -> bool:
    """Return True if the path points to a supported image file."""
    try:
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS
    except Exception:
        return False


def encode_image_to_data_uri(path: Union[str, Path]) -> str:
    """Encode an image file as a base64 data URI."""
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _image_block_to_data_uri(block: Dict[str, Any]) -> Optional[str]:
    """Extract a data URI from an OpenAI-style image_url content block."""
    if block.get("type") != "image_url":
        return None
    image_url = block.get("image_url", {})
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return image_url.get("url", "")
    return None


def _data_uri_to_path(data_uri: str) -> Optional[Path]:
    """If the data URI came from a local file, try to recover the path."""
    if data_uri.startswith("data:"):
        return None
    p = Path(data_uri)
    if p.exists() and p.is_file():
        return p
    return None


def content_has_image_blocks(content: Union[str, List[Dict[str, Any]]]) -> bool:
    """Return True if content contains image blocks."""
    if isinstance(content, str):
        return False
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in content
        )
    return False


def extract_image_paths_from_content(
    content: Union[str, List[Dict[str, Any]]]
) -> List[Path]:
    """Return local image paths referenced by content blocks, if any."""
    paths: List[Path] = []
    if isinstance(content, str):
        return paths
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            data_uri = _image_block_to_data_uri(block)
            if data_uri:
                path = _data_uri_to_path(data_uri)
                if path:
                    paths.append(path)
    return paths


def adapt_content_for_non_vision_model(
    content: Union[str, List[Dict[str, Any]]],
) -> str:
    """Replace image blocks in content with placeholder text.

    Returns a plain string suitable for models that do not accept image input.
    The actual images are tracked separately in the context container.
    """
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue

        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif btype == "image_url":
            data_uri = _image_block_to_data_uri(block)
            if not data_uri:
                continue
            path = _data_uri_to_path(data_uri)
            label = str(path) if path else f"attached image {i + 1}"
            parts.append(
                f"[Image: {label} — attached image omitted because the active model "
                "does not support vision.]"
            )

    return "\n\n".join(parts)


def filter_messages_for_non_vision_model(
    messages: List[Any],
) -> List[Any]:
    """Return a new message list with image blocks stripped from all messages.

    Original messages are not modified; this creates filtered copies for
    non-vision API calls while preserving full history for context.
    """
    filtered = []
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else msg
        if isinstance(content, str):
            # String content: keep as-is
            filtered.append(msg)
        elif isinstance(content, list):
            # Multimodal content: filter out image blocks
            filtered_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    # Skip image block
                    continue
                filtered_blocks.append(block)
            # Create a new message object with filtered content
            if filtered_blocks:
                # Preserve all other attributes of the message
                filtered_content = filtered_blocks
                if hasattr(msg, 'copy'):
                    new_msg = msg.copy()
                    new_msg.content = filtered_content
                    filtered.append(new_msg)
                else:
                    filtered.append(filtered_content)
            else:
                # No content left after filtering, add a note
                if hasattr(msg, 'copy'):
                    new_msg = msg.copy()
                    new_msg.content = "[Images omitted for non-vision model]"
                    filtered.append(new_msg)
                else:
                    filtered.append("[Images omitted for non-vision model]")
        else:
            filtered.append(msg)
    return filtered
