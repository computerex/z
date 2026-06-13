"""Probe what the live provider returns for a trivial message."""
import os, sys, asyncio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness.config import Config
from harness.streaming_client import StreamingJSONClient, StreamingMessage
from harness.tool_registry import tool_defs_to_openai_tools


async def main():
    cfg = Config.from_json(workspace=ROOT)
    client = StreamingJSONClient(
        api_key=cfg.api_key, base_url=cfg.api_url, model=cfg.model,
        temperature=cfg.temperature, max_tokens=4096,
    )
    client.reasoning_effort = "high"
    tools = tool_defs_to_openai_tools()

    contents = []
    reasonings = []
    async with client:
        resp = await client.chat_stream_raw(
            messages=[
                StreamingMessage(role="system", content="You are a helpful assistant."),
                StreamingMessage(role="user", content="hi"),
            ],
            on_content=lambda c: contents.append(c),
            on_reasoning=lambda c: reasonings.append(c),
            tools=tools,
        )
    print("=== RESULT ===")
    print(f"resp.content      = {resp.content!r}")
    print(f"resp.thinking     = {(resp.thinking or '')[:120]!r}")
    print(f"resp.finish       = {resp.finish_reason!r}")
    print(f"resp.tool_calls   = {resp.tool_calls!r}")
    print(f"streamed content  = {''.join(contents)!r}")
    print(f"streamed reasoning_len = {len(''.join(reasonings))}")


if __name__ == "__main__":
    asyncio.run(main())
