"""Tests for the Codex SSE reader's handling of oversized event lines.

aiohttp's line iterator caps a single SSE line at 131072 bytes, which Codex
`response.completed` events can exceed. The client now splits lines from raw
chunks, so this test feeds the same parsing logic a >128KB event and asserts
it parses.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_oversized_sse_line_parses():
    """Simulate the chunk-splitting loop used by codex_oauth_client.chat_stream."""
    big_summary = "x" * 300_000  # > 131072 bytes inside one event
    event = {
        "type": "response.completed",
        "response": {"id": "resp_test", "reasoning": {"summary": big_summary}},
    }
    stream = (
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n'
        + b"data: "
        + json.dumps(event).encode()
        + b"\n"
        + b"data: [DONE]\n"
    )

    # Feed it in small chunks like a real socket would.
    chunks = [stream[i : i + 4096] for i in range(0, len(stream), 4096)]

    events = []
    buf = b""

    def feed(chunk: bytes):
        nonlocal buf
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                return True
            events.append(json.loads(data))
        return False

    done = False
    for chunk in chunks:
        if feed(chunk):
            done = True
            break

    assert done
    assert len(events) == 2
    assert events[0] == {"type": "response.output_text.delta", "delta": "hi"}
    assert events[1]["type"] == "response.completed"
    assert len(events[1]["response"]["reasoning"]["summary"]) == 300_000
