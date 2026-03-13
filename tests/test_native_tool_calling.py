"""Headless test for native OpenAI-compatible tool calling via Ollama.

Verifies that:
1. Ollama model fetching works (fixed /api/tags URL)
2. Tool schemas are generated correctly
3. Native tool calls are parsed from SSE stream
4. End-to-end: model receives tools and returns a function call
"""

import asyncio
import json
import sys
import os
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.streaming_client import StreamingJSONClient, StreamingMessage
from harness.tool_registry import tools_to_openai_schema

OLLAMA_URL = "http://localhost:11434/v1/"
API_KEY = "foo"
MODELS_TO_TEST = ["gpt-oss:20b", "qwen3.5:9b"]

MINIMAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {"type": "string", "description": "true for recursive"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["command"],
            },
        },
    },
]


def warm_model(model: str):
    """Send a tiny request to ensure the model is loaded in Ollama.
    
    Ollama may need 30-90s to unload the previous model and load a new one,
    especially for large models. Retry a few times with generous timeouts.
    """
    print(f"  Warming up {model}...")
    for attempt in range(3):
        try:
            r = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=300.0,
            )
            r.raise_for_status()
            print(f"  {model} loaded OK")
            return
        except Exception as e:
            print(f"  Warm-up attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                import time
                time.sleep(5)
    print(f"  Warning: could not warm {model}, proceeding anyway")


def test_tool_schema_generation():
    """Verify OpenAI tool schemas are well-formed."""
    schemas = tools_to_openai_schema()
    assert len(schemas) > 0, "No tool schemas generated"
    for s in schemas:
        assert s["type"] == "function", f"Bad type: {s.get('type')}"
        func = s["function"]
        assert "name" in func, f"Missing name in {func}"
        assert "description" in func, f"Missing description for {func['name']}"
    names = [s["function"]["name"] for s in schemas]
    assert "read_file" in names
    assert "execute_command" in names
    assert "replace_in_file" in names
    print(f"  [PASS] Schema generation: {len(schemas)} tools")
    return schemas


def test_ollama_model_fetching():
    """Verify the fixed Ollama model fetch URL works."""
    import importlib.util
    harness_file = os.path.join(os.path.dirname(__file__), "..", "harness.py")
    spec = importlib.util.spec_from_file_location("root_harness", harness_file)
    root_harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_harness)
    models = root_harness._fetch_models_ollama(OLLAMA_URL, API_KEY)
    assert len(models) > 0, "No models returned from Ollama"
    print(f"  [PASS] Model fetching: {len(models)} models — {models[:5]}")
    return models


async def test_native_tool_call(model: str, tools: list):
    """Send a message that should trigger a tool call, verify we get one back."""
    client = StreamingJSONClient(
        api_key=API_KEY,
        base_url=OLLAMA_URL,
        model=model,
        temperature=0.0,
        max_tokens=4096,
    )

    messages = [
        StreamingMessage(
            role="system",
            content=(
                "You are a coding assistant. You have tools available. "
                "When asked to read a file, call the read_file tool. "
                "When asked to list files, call the list_files tool. "
                "When asked to run a command, call the execute_command tool. "
                "Always use tools — never just describe what you would do."
            ),
        ),
        StreamingMessage(
            role="user",
            content="Read the file README.md",
        ),
    ]

    content_chunks = []
    reasoning_chunks = []

    async with client:
        response = await client.chat_stream_raw(
            messages=messages,
            on_content=lambda c: content_chunks.append(c),
            on_reasoning=lambda c: reasoning_chunks.append(c),
            tools=tools,
            max_retries=5,
        )

    reasoning_text = "".join(reasoning_chunks)
    print(f"\n  Model: {model}")
    print(f"  finish_reason: {response.finish_reason}")
    print(f"  content length: {len(response.content or '')}")
    if reasoning_text:
        print(f"  reasoning: {reasoning_text[:200]}")
    if response.content:
        print(f"  content preview: {(response.content or '')[:200]}")
    print(f"  native_tool_calls: {len(response.native_tool_calls)}")
    for tc in response.native_tool_calls:
        print(f"    -> {tc.name}({json.dumps(tc.parameters)})")

    assert len(response.native_tool_calls) > 0, (
        f"Expected at least one native tool call, got none. "
        f"Content: {(response.content or '')[:300]}"
    )
    tc = response.native_tool_calls[0]
    assert tc.name == "read_file", f"Expected read_file, got {tc.name}"
    assert "path" in tc.parameters, f"Missing 'path' param: {tc.parameters}"
    print(f"  [PASS] Native tool call: {tc.name}(path={tc.parameters.get('path')})")
    return tc


async def test_full_schema_tool_call(model: str):
    """Test with the full harness tool schema (all 23 tools)."""
    schemas = tools_to_openai_schema()
    client = StreamingJSONClient(
        api_key=API_KEY,
        base_url=OLLAMA_URL,
        model=model,
        temperature=0.0,
        max_tokens=4096,
    )

    messages = [
        StreamingMessage(
            role="system",
            content=(
                "You are a coding assistant with tools available. "
                "Use the appropriate tool for each request. "
                "Always respond with a tool call, not text."
            ),
        ),
        StreamingMessage(
            role="user",
            content="List the files in the src/ directory",
        ),
    ]

    async with client:
        response = await client.chat_stream_raw(
            messages=messages, tools=schemas, max_retries=5,
        )

    print(f"\n  Full schema test — {model}")
    print(f"  finish_reason: {response.finish_reason}")
    print(f"  native_tool_calls: {len(response.native_tool_calls)}")
    for tc in response.native_tool_calls:
        print(f"    -> {tc.name}({json.dumps(tc.parameters)})")

    assert len(response.native_tool_calls) > 0, (
        f"Full schema: no tool call. Content: {(response.content or '')[:300]}"
    )
    print(f"  [PASS] Full schema tool call: {response.native_tool_calls[0].name}")


async def run_all():
    print("=" * 60)
    print("Native Tool Calling Test Suite")
    print("=" * 60)

    print("\n1. Tool schema generation")
    test_tool_schema_generation()

    print("\n2. Ollama model fetching (fixed URL)")
    test_ollama_model_fetching()

    for model in MODELS_TO_TEST:
        warm_model(model)

        print(f"\n3. Minimal tool call test — {model}")
        await test_native_tool_call(model, MINIMAL_TOOLS)

        print(f"\n4. Full schema tool call test — {model}")
        await test_full_schema_tool_call(model)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all())
