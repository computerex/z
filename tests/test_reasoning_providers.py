"""Cross-provider reasoning/thinking integration test.

Tests that the streaming client correctly extracts reasoning tokens
across different providers, models, and streaming paths (raw SSE vs LiteLLM).

Run:  python tests/test_reasoning_providers.py
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.streaming_client import StreamingJSONClient, StreamingMessage

CONFIG_PATH = os.path.expanduser("~/.z.json")
PROMPT = "What is 2+2? Think step by step but keep your answer to one line."


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ── provider definitions ──────────────────────────────────────────────
# Each entry: (label, profile_name, model_override, expect_reasoning)
# expect_reasoning: True = model should produce reasoning tokens
#                   False = model doesn't support reasoning (baseline check)

PROVIDER_TESTS = [
    # NanoGPT — kimi-k2.5:thinking  (raw SSE, `reasoning` field)
    ("NanoGPT/kimi-k2.5:thinking", "nanogpt", "moonshotai/kimi-k2.5:thinking", True),
    # NanoGPT — deepseek-r1  (raw SSE, `reasoning_content` field)
    ("NanoGPT/deepseek-r1", "nanogpt", "deepseek/deepseek-r1", True),
    # NanoGPT — QwQ-32B (raw SSE, likely `reasoning` or think tags)
    ("NanoGPT/QwQ-32B", "nanogpt", "Qwen/QwQ-32B", True),
    # NanoGPT — non-reasoning model baseline
    ("NanoGPT/kimi-k2.5", "nanogpt", "moonshotai/kimi-k2.5", False),
    # Bedrock — qwen3 (LiteLLM path, no separate reasoning tokens)
    ("Bedrock/qwen3", "bedrock", None, False),
    # ZAI — glm-4.7 (LiteLLM path, native thinking)
    ("ZAI/glm-4.7", "zai-coding", None, False),
]


async def test_provider(label, profile, model_override, expect_reasoning, cfg):
    """Test a single provider/model combo for reasoning extraction."""
    providers = cfg.get("providers", {})
    if profile not in providers:
        return {"label": label, "status": "SKIP", "reason": f"profile '{profile}' not configured"}

    pcfg = providers[profile]
    api_url = pcfg.get("api_url", "")
    api_key = pcfg.get("api_key", "")
    model = model_override or pcfg.get("model", "")

    if not api_key:
        return {"label": label, "status": "SKIP", "reason": "no API key"}

    # Collect streaming data
    content_chunks = []
    reasoning_chunks = []

    def on_content(text):
        content_chunks.append(text)

    def on_reasoning(text):
        reasoning_chunks.append(text)

    t0 = time.time()
    try:
        async with StreamingJSONClient(
            api_key=api_key,
            base_url=api_url,
            model=model,
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
        ) as client:
            response = await asyncio.wait_for(
                client.chat_stream_raw(
                    messages=[StreamingMessage(role="user", content=PROMPT)],
                    on_content=on_content,
                    on_reasoning=on_reasoning,
                ),
                timeout=45.0,
            )
    except asyncio.TimeoutError:
        return {"label": label, "status": "FAIL", "reason": "timeout (45s)", "elapsed": time.time() - t0}
    except Exception as e:
        return {"label": label, "status": "FAIL", "reason": str(e)[:120], "elapsed": time.time() - t0}

    elapsed = time.time() - t0
    full_content = "".join(content_chunks)
    full_reasoning = "".join(reasoning_chunks)

    # Also check response.thinking for reasoning that came via the response object
    resp_thinking = response.thinking or ""

    has_reasoning_callback = len(full_reasoning) > 0
    has_reasoning_response = len(resp_thinking) > 0
    has_think_tags = "<think>" in full_content.lower()
    has_content = len(full_content.strip()) > 0

    # Determine result
    reasoning_detected = has_reasoning_callback or has_reasoning_response or has_think_tags

    if not has_content:
        status = "FAIL"
        reason = "no content returned"
    elif expect_reasoning and not reasoning_detected:
        status = "WARN"
        reason = "expected reasoning but none detected"
    elif not expect_reasoning and reasoning_detected:
        status = "INFO"
        reason = "unexpected reasoning detected (bonus!)"
    else:
        status = "PASS"
        reason = ""

    return {
        "label": label,
        "status": status,
        "reason": reason,
        "elapsed": round(elapsed, 1),
        "content_len": len(full_content),
        "reasoning_len": len(full_reasoning),
        "reasoning_chunks": len(reasoning_chunks),
        "content_chunks": len(content_chunks),
        "has_think_tags": has_think_tags,
        "content_preview": full_content[:80].replace("\n", " "),
        "reasoning_preview": full_reasoning[:80].replace("\n", " "),
        "resp_thinking_len": len(resp_thinking),
    }


async def main():
    cfg = load_config()

    print("=" * 78)
    print("  Cross-Provider Reasoning Test")
    print("=" * 78)
    print(f"  Prompt: {PROMPT!r}")
    print()

    results = []
    for label, profile, model_override, expect_reasoning in PROVIDER_TESTS:
        print(f"  Testing {label} ...", end="", flush=True)
        result = await test_provider(label, profile, model_override, expect_reasoning, cfg)
        results.append(result)

        status = result["status"]
        symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "—", "INFO": "ℹ"}[status]
        elapsed = result.get("elapsed", 0)
        reason = result.get("reason", "")

        print(f"\r  {symbol} {label:40s}", end="")
        if status == "SKIP":
            print(f"  SKIP  {reason}")
        elif status == "FAIL":
            print(f"  FAIL  {reason}  ({elapsed}s)")
        elif status == "WARN":
            print(f"  WARN  {reason}  ({elapsed}s)")
        else:
            r_chunks = result.get("reasoning_chunks", 0)
            r_len = result.get("reasoning_len", 0)
            c_len = result.get("content_len", 0)
            think_tags = " [<think> in content]" if result.get("has_think_tags") else ""
            reasoning_info = f"reasoning={r_len}ch/{r_chunks}chunks" if r_len else "no reasoning"
            print(f"  PASS  content={c_len}ch  {reasoning_info}{think_tags}  ({elapsed}s)")

    # ── Summary ──
    print()
    print("-" * 78)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warned = sum(1 for r in results if r["status"] == "WARN")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    info = sum(1 for r in results if r["status"] == "INFO")

    print(f"  Results: {passed} passed, {failed} failed, {warned} warnings, {skipped} skipped, {info} info")

    # Detail for reasoning models
    print()
    print("  Reasoning extraction details:")
    for r in results:
        if r["status"] in ("SKIP",):
            continue
        label = r["label"]
        rlen = r.get("reasoning_len", 0)
        rchunks = r.get("reasoning_chunks", 0)
        rprev = r.get("reasoning_preview", "")
        cprev = r.get("content_preview", "")
        think = r.get("has_think_tags", False)
        resp_t = r.get("resp_thinking_len", 0)

        print(f"\n  {label}:")
        if rlen:
            print(f"    Reasoning ({rlen} chars, {rchunks} chunks): {rprev!r}")
        if think:
            print(f"    <think> tags found in content stream")
        if resp_t and not rlen:
            print(f"    response.thinking present ({resp_t} chars)")
        if not rlen and not think and not resp_t:
            print(f"    No reasoning detected")
        print(f"    Content ({r.get('content_len', 0)} chars): {cprev!r}")

    print()
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
