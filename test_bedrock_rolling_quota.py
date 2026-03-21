"""Precise investigation of Bedrock token quota behavior.

Findings so far:
- Error says "Too many tokens per day" but it's NOT a fixed daily reset
- Hit limits at 5pm, partially recovered by 9:50pm (4.75 hours later)
- This suggests a ROLLING WINDOW token quota

This script measures:
1. How many tokens we can use right now before hitting the limit
2. The exact token count consumed per request
3. How quickly tokens "refill" (probing at intervals)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load config
config_path = Path.home() / ".z.json"
config = json.loads(config_path.read_text())
bedrock_cfg = config.get("providers", {}).get("bedrock", {})

API_KEY = bedrock_cfg.get("api_key", "")
REGION = "us-east-1"

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = API_KEY

session = boto3.Session()
runtime = session.client(
    "bedrock-runtime",
    region_name=REGION,
    config=BotoConfig(retries={"max_attempts": 0}, read_timeout=60),
)


def try_request(model_id, prompt="Say OK.", max_tokens=5):
    """Send a minimal request. Returns (success, input_tokens, output_tokens, error_msg)."""
    try:
        resp = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": "Be brief."}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
        )
        text = ""
        for b in resp.get("output", {}).get("message", {}).get("content", []):
            text += b.get("text", "")
        usage = resp.get("usage", {})
        inp = usage.get("inputTokens", 0)
        out = usage.get("outputTokens", 0)
        return True, inp, out, text
    except ClientError as e:
        msg = e.response["Error"]["Message"]
        code = e.response["Error"]["Code"]
        return False, 0, 0, f"{code}: {msg}"
    except Exception as e:
        return False, 0, 0, f"{type(e).__name__}: {e}"


def measure_available_tokens(model_id, label=""):
    """Send requests until throttled, tracking total tokens consumed."""
    print(f"\n--- Measuring available tokens for {label or model_id} ---")
    print(f"    Time: {datetime.now().strftime('%H:%M:%S')}")
    
    total_input = 0
    total_output = 0
    total_requests = 0
    
    # Start with a tiny request to check if we're already throttled
    ok, inp, out, detail = try_request(model_id, "Hi", max_tokens=5)
    if not ok:
        print(f"    Already throttled: {detail}")
        return 0, 0, 0
    
    total_input += inp
    total_output += out
    total_requests += 1
    print(f"    [1] OK: {inp}in/{out}out (total: {total_input+total_output} tokens)")
    
    # Now send progressively larger requests to measure quota
    prompts = [
        ("Count from 1 to 10.", 50),
        ("Count from 1 to 20.", 100),
        ("Write a haiku about the ocean.", 50),
        ("Explain what Python is in 2 sentences.", 100),
        ("List 5 prime numbers.", 50),
        ("What is 2+2? Explain step by step.", 100),
        ("Say hello world in 3 programming languages.", 150),
        ("Write a short poem about coding.", 100),
        ("What are the 4 seasons?", 50),
        ("Name 3 planets.", 30),
        ("Say bye.", 10),
        ("Say yes.", 5),
        ("Say no.", 5),
        ("Hi.", 5),
        ("OK.", 5),
    ]
    
    for prompt, max_tok in prompts:
        ok, inp, out, detail = try_request(model_id, prompt, max_tok)
        total_requests += 1
        
        if ok:
            total_input += inp
            total_output += out
            print(f"    [{total_requests}] OK: {inp}in/{out}out (cumulative: {total_input+total_output} tokens)")
        else:
            print(f"    [{total_requests}] THROTTLED after {total_input+total_output} total tokens")
            print(f"        Error: {detail[:80]}")
            break
        
        time.sleep(0.5)  # Small delay to be fair
    
    print(f"\n    Summary: {total_requests} requests, {total_input} input + {total_output} output = {total_input+total_output} total tokens")
    return total_input, total_output, total_requests


def probe_refill_rate(model_id, interval_seconds=60, num_probes=30, label=""):
    """Probe at regular intervals to see when tokens become available again."""
    print(f"\n--- Probing refill rate for {label or model_id} ---")
    print(f"    Probing every {interval_seconds}s for up to {num_probes} probes")
    print(f"    Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    for i in range(num_probes):
        time.sleep(interval_seconds)
        
        ok, inp, out, detail = try_request(model_id, "Hi", max_tokens=5)
        now = datetime.now().strftime('%H:%M:%S')
        elapsed = timedelta(seconds=(i + 1) * interval_seconds)
        
        if ok:
            print(f"    [{now}] (+{elapsed}) ✓ Available! {inp}in/{out}out")
            # Now measure how many tokens we have
            measure_available_tokens(model_id, label)
            return (i + 1) * interval_seconds
        else:
            print(f"    [{now}] (+{elapsed}) ✗ Still throttled")
    
    print(f"    Did not recover within {num_probes * interval_seconds}s")
    return None


def test_all_profiles():
    """Test current status of all Opus profiles."""
    print("\n=== Current Status of All Opus Profiles ===")
    print(f"    Time: {datetime.now().strftime('%H:%M:%S')}")
    
    models = [
        ("us.anthropic.claude-opus-4-6-v1", "Opus 4.6 US"),
        ("global.anthropic.claude-opus-4-6-v1", "Opus 4.6 Global"),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", "Opus 4.5 US"),
        ("global.anthropic.claude-opus-4-5-20251101-v1:0", "Opus 4.5 Global"),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "Opus 4.1 US"),
        ("us.anthropic.claude-sonnet-4-6", "Sonnet 4.6 US"),
        ("global.anthropic.claude-sonnet-4-6", "Sonnet 4.6 Global"),
    ]
    
    for model_id, desc in models:
        ok, inp, out, detail = try_request(model_id, "Say OK.", max_tokens=5)
        status = "✓" if ok else "✗"
        if ok:
            print(f"    {status} {desc:25s} — OK ({inp}in/{out}out)")
        else:
            print(f"    {status} {desc:25s} — {detail[:60]}")
        time.sleep(0.5)


def measure_and_track():
    """Main flow: check status, measure available tokens, probe refill."""
    print("=" * 70)
    print(f"Bedrock Rolling Quota Investigation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Check current status of all profiles
    test_all_profiles()
    
    # Step 2: For each available profile, measure how many tokens we can use
    print("\n" + "=" * 70)
    print("Measuring available token budget per profile...")
    print("=" * 70)
    
    profiles = [
        ("us.anthropic.claude-opus-4-6-v1", "Opus 4.6 US"),
        ("global.anthropic.claude-opus-4-6-v1", "Opus 4.6 Global"),
    ]
    
    for model_id, desc in profiles:
        ok, _, _, _ = try_request(model_id, "Hi", max_tokens=5)
        if ok:
            measure_available_tokens(model_id, desc)
        else:
            print(f"\n--- {desc} already throttled, will probe refill ---")
    
    # Step 3: If both are throttled, probe refill rate
    us_ok, _, _, _ = try_request("us.anthropic.claude-opus-4-6-v1", "Hi", 5)
    gl_ok, _, _, _ = try_request("global.anthropic.claude-opus-4-6-v1", "Hi", 5)
    
    if not us_ok and not gl_ok:
        print("\n" + "=" * 70)
        print("Both profiles throttled. Probing refill rate...")
        print("(Checking every 60s)")
        print("=" * 70)
        probe_refill_rate("us.anthropic.claude-opus-4-6-v1", 
                         interval_seconds=60, num_probes=60, label="Opus 4.6 US")
    elif not us_ok:
        print("\n" + "=" * 70)
        print("US profile throttled. Probing refill rate...")
        print("(Checking every 60s)")
        print("=" * 70)
        probe_refill_rate("us.anthropic.claude-opus-4-6-v1",
                         interval_seconds=60, num_probes=60, label="Opus 4.6 US")
    
    print("\n" + "=" * 70)
    print(f"Investigation complete at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    measure_and_track()