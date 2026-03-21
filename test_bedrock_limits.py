"""Test program to investigate Claude Opus 4.6 throttling/quota limits on AWS Bedrock.

This script sends a series of small requests to Bedrock and measures:
- Time between successful requests
- Exact error messages when throttled
- How long until requests succeed again after throttling
- Token limits per minute / per request
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

# Load config
config_path = Path.home() / ".z.json"
config = json.loads(config_path.read_text())
bedrock_cfg = config.get("providers", {}).get("bedrock", {})

API_KEY = bedrock_cfg.get("api_key", "")
MODEL = bedrock_cfg.get("model", "us.anthropic.claude-opus-4-6-v1")
REGION = "us-east-1"  # extracted from api_url

print(f"=" * 70)
print(f"Bedrock Throttle/Quota Investigation")
print(f"=" * 70)
print(f"Model:  {MODEL}")
print(f"Region: {REGION}")
print(f"Time:   {datetime.now().isoformat()}")
print(f"=" * 70)

# Setup boto3
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# Set bearer token
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = API_KEY

session = boto3.Session()
client = session.client(
    "bedrock-runtime",
    region_name=REGION,
    config=BotoConfig(
        retries={"max_attempts": 0},  # Disable boto3 auto-retry so we see raw errors
        read_timeout=60,
        connect_timeout=10,
    ),
)

# Also get the management client for quota info
try:
    bedrock_mgmt = session.client("bedrock", region_name=REGION)
except Exception as e:
    print(f"[WARN] Could not create bedrock management client: {e}")
    bedrock_mgmt = None


def get_model_info():
    """Try to get model/quota info from Bedrock management API."""
    if not bedrock_mgmt:
        return
    
    print(f"\n--- Model / Quota Info ---")
    
    # Try to get foundation model info
    try:
        resp = bedrock_mgmt.get_foundation_model(modelIdentifier=MODEL)
        model_details = resp.get("modelDetails", {})
        print(f"  Model ID:        {model_details.get('modelId', 'N/A')}")
        print(f"  Provider:        {model_details.get('providerName', 'N/A')}")
        print(f"  Input Modalities: {model_details.get('inputModalities', [])}")
        print(f"  Output Modalities: {model_details.get('outputModalities', [])}")
        print(f"  Streaming:       {model_details.get('responseStreamingSupported', 'N/A')}")
        print(f"  Inference Types: {model_details.get('inferenceTypesSupported', [])}")
    except Exception as e:
        print(f"  get_foundation_model error: {e}")
    
    # Try to list inference profiles
    try:
        resp = bedrock_mgmt.list_inference_profiles()
        profiles = resp.get("inferenceProfileSummaries", [])
        print(f"\n  Inference Profiles ({len(profiles)} total):")
        for p in profiles:
            pid = p.get("inferenceProfileId", "")
            if "claude" in pid.lower() and ("opus" in pid.lower() or "4-6" in pid or "4.6" in pid):
                print(f"    - {pid} (status: {p.get('status')}, type: {p.get('type')})")
    except Exception as e:
        print(f"  list_inference_profiles error: {e}")

    print()


def send_small_request(prompt="Say hi in exactly 3 words.", max_tokens=50):
    """Send a minimal request and return (success, response_text, error, duration, input_tokens, output_tokens)."""
    system_prompt = [{"text": "You are a helpful assistant. Be very brief."}]
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]
    
    request = {
        "modelId": MODEL,
        "messages": messages,
        "system": system_prompt,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.1,
        },
    }
    
    t0 = time.time()
    try:
        response = client.converse(**request)
        duration = time.time() - t0
        
        # Extract response
        output = response.get("output", {})
        msg = output.get("message", {})
        content_blocks = msg.get("content", [])
        text = " ".join(b.get("text", "") for b in content_blocks)
        
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        
        stop_reason = response.get("stopReason", "unknown")
        
        return {
            "success": True,
            "text": text,
            "error": None,
            "duration": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
        }
    except ClientError as e:
        duration = time.time() - t0
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        http_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        
        # Check for retry-after header
        headers = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        retry_after = headers.get("retry-after", headers.get("x-retry-after", "N/A"))
        
        return {
            "success": False,
            "text": None,
            "error": f"{error_code} (HTTP {http_code}): {error_msg}",
            "error_code": error_code,
            "http_code": http_code,
            "retry_after": retry_after,
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
            "headers": dict(headers),
        }
    except Exception as e:
        duration = time.time() - t0
        return {
            "success": False,
            "text": None,
            "error": f"{type(e).__name__}: {e}",
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def send_streaming_request(prompt="Say hi in exactly 3 words.", max_tokens=50):
    """Send a minimal streaming request to test converse_stream limits."""
    system_prompt = [{"text": "You are a helpful assistant. Be very brief."}]
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]
    
    request = {
        "modelId": MODEL,
        "messages": messages,
        "system": system_prompt,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.1,
        },
    }
    
    t0 = time.time()
    try:
        response = client.converse_stream(**request)
        stream = response.get("stream", [])
        
        text_parts = []
        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"
        first_chunk_time = None
        
        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                t = delta.get("text", "")
                if t:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - t0
                    text_parts.append(t)
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "unknown")
        
        duration = time.time() - t0
        return {
            "success": True,
            "text": "".join(text_parts),
            "error": None,
            "duration": duration,
            "first_chunk_time": first_chunk_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
        }
    except ClientError as e:
        duration = time.time() - t0
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        http_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        headers = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        retry_after = headers.get("retry-after", headers.get("x-retry-after", "N/A"))
        
        return {
            "success": False,
            "text": None,
            "error": f"{error_code} (HTTP {http_code}): {error_msg}",
            "error_code": error_code,
            "http_code": http_code,
            "retry_after": retry_after,
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
            "headers": dict(headers),
        }
    except Exception as e:
        duration = time.time() - t0
        return {
            "success": False,
            "text": None,
            "error": f"{type(e).__name__}: {e}",
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def test_initial_request():
    """Test 1: Can we make a single request?"""
    print("\n" + "=" * 70)
    print("TEST 1: Single request (non-streaming)")
    print("=" * 70)
    
    result = send_small_request()
    if result["success"]:
        print(f"  ✓ SUCCESS in {result['duration']:.2f}s")
        print(f"    Response: {result['text']}")
        print(f"    Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"    Stop reason: {result['stop_reason']}")
    else:
        print(f"  ✗ FAILED in {result['duration']:.2f}s")
        print(f"    Error: {result['error']}")
        if "headers" in result:
            print(f"    Retry-After: {result.get('retry_after', 'N/A')}")
            # Print all headers for debugging
            print(f"    Response headers:")
            for k, v in result.get("headers", {}).items():
                print(f"      {k}: {v}")
    return result


def test_streaming_request():
    """Test 2: Can we make a single streaming request?"""
    print("\n" + "=" * 70)
    print("TEST 2: Single request (streaming)")
    print("=" * 70)
    
    result = send_streaming_request()
    if result["success"]:
        print(f"  ✓ SUCCESS in {result['duration']:.2f}s")
        print(f"    Response: {result['text']}")
        print(f"    First chunk at: {result.get('first_chunk_time', 0):.2f}s")
        print(f"    Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    else:
        print(f"  ✗ FAILED in {result['duration']:.2f}s")
        print(f"    Error: {result['error']}")
        if "headers" in result:
            print(f"    Retry-After: {result.get('retry_after', 'N/A')}")
            print(f"    Response headers:")
            for k, v in result.get("headers", {}).items():
                print(f"      {k}: {v}")
    return result


def test_burst_rate():
    """Test 3: How many requests can we send in a burst before throttling?"""
    print("\n" + "=" * 70)
    print("TEST 3: Burst rate test (rapid sequential requests)")
    print("=" * 70)
    
    results = []
    for i in range(10):
        t0 = time.time()
        result = send_small_request(f"Say the number {i+1}.", max_tokens=10)
        results.append(result)
        
        status = "✓" if result["success"] else "✗"
        detail = result["text"] if result["success"] else result["error"][:80]
        print(f"  [{i+1:2d}] {status} {result['duration']:.2f}s | {detail}")
        
        if not result["success"]:
            # If throttled, note it and stop burst
            print(f"       Throttled after {i+1} requests")
            break
    
    successes = sum(1 for r in results if r["success"])
    print(f"\n  Summary: {successes}/{len(results)} succeeded")
    return results


def test_throttle_recovery():
    """Test 4: After getting throttled, how long until we can send again?"""
    print("\n" + "=" * 70)
    print("TEST 4: Throttle recovery timing")
    print("=" * 70)
    
    # First, trigger throttling with a burst
    print("  Phase 1: Triggering throttle...")
    throttled = False
    for i in range(15):
        result = send_small_request(f"Number {i}.", max_tokens=10)
        if not result["success"]:
            print(f"  Throttled after {i+1} requests: {result['error'][:80]}")
            throttled = True
            break
        else:
            print(f"    Request {i+1} OK ({result['duration']:.2f}s)")
    
    if not throttled:
        print("  Could not trigger throttling after 15 requests!")
        return
    
    # Now probe at increasing intervals
    print("\n  Phase 2: Probing recovery time...")
    wait_times = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300]
    
    for wait in wait_times:
        print(f"\n  Waiting {wait}s...", end="", flush=True)
        time.sleep(wait)
        
        result = send_small_request("Say OK.", max_tokens=5)
        if result["success"]:
            print(f" ✓ RECOVERED! (took {wait}s)")
            print(f"    Response: {result['text']}")
            
            # Now test: can we do a second request immediately?
            result2 = send_small_request("Say YES.", max_tokens=5)
            if result2["success"]:
                print(f"    Follow-up also succeeded ({result2['duration']:.2f}s)")
            else:
                print(f"    Follow-up throttled again: {result2['error'][:60]}")
            return wait
        else:
            print(f" ✗ Still throttled: {result['error'][:60]}")
    
    print("\n  ✗ Did not recover within 5 minutes!")
    return None


def test_token_quota():
    """Test 5: Test if it's token-based throttling (large vs small requests)."""
    print("\n" + "=" * 70)
    print("TEST 5: Token-based quota test")
    print("=" * 70)
    
    # First try a minimal request
    print("  Sending minimal request (5 tokens)...")
    r1 = send_small_request("Hi", max_tokens=5)
    if r1["success"]:
        print(f"  ✓ Minimal: {r1['input_tokens']} in / {r1['output_tokens']} out ({r1['duration']:.2f}s)")
    else:
        print(f"  ✗ Even minimal request failed: {r1['error'][:80]}")
        return
    
    # Try a larger request
    print("  Sending larger request (500 tokens)...")
    r2 = send_small_request(
        "Write a detailed paragraph about the history of computing, covering at least 5 key milestones.",
        max_tokens=500
    )
    if r2["success"]:
        print(f"  ✓ Larger: {r2['input_tokens']} in / {r2['output_tokens']} out ({r2['duration']:.2f}s)")
    else:
        print(f"  ✗ Larger request failed: {r2['error'][:80]}")
    
    # Try another immediately
    print("  Sending another large request immediately...")
    r3 = send_small_request(
        "Write a detailed paragraph about the history of artificial intelligence.",
        max_tokens=500
    )
    if r3["success"]:
        print(f"  ✓ Second large: {r3['input_tokens']} in / {r3['output_tokens']} out ({r3['duration']:.2f}s)")
    else:
        print(f"  ✗ Second large request failed: {r3['error'][:80]}")


def test_model_variants():
    """Test 6: Try different model ID formats to see which work."""
    print("\n" + "=" * 70)
    print("TEST 6: Model ID variant test")
    print("=" * 70)
    
    variants = [
        "us.anthropic.claude-opus-4-6-v1",
        "anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-6-v1:0",
        "anthropic.claude-opus-4-6-v1:0",
    ]
    
    for model_id in variants:
        global MODEL
        old_model = MODEL
        MODEL = model_id
        
        result = send_small_request("Say OK.", max_tokens=5)
        status = "✓" if result["success"] else "✗"
        detail = result["text"] if result["success"] else result["error"][:60]
        print(f"  {status} {model_id}")
        print(f"    {detail} ({result['duration']:.2f}s)")
        
        MODEL = old_model
        
        # Don't burn requests if first ones fail
        if not result["success"] and "throttl" in result.get("error", "").lower():
            print("  (Stopping variant test - throttled)")
            break


if __name__ == "__main__":
    print("\nStarting Bedrock limits investigation...\n")
    
    # Get model info first
    get_model_info()
    
    # Test 1: Basic connectivity
    r1 = test_initial_request()
    
    if r1["success"]:
        # Test 2: Streaming
        test_streaming_request()
        
        # Test 3: Burst rate
        test_burst_rate()
        
        # Small pause before recovery test
        print("\n  Pausing 5s before recovery test...")
        time.sleep(5)
        
        # Test 4: Recovery timing
        test_throttle_recovery()
    else:
        # Already throttled - go straight to recovery test
        print("\n  Already throttled! Skipping to recovery test...")
        
        # Test recovery
        test_throttle_recovery()
    
    print("\n" + "=" * 70)
    print("Investigation complete.")
    print("=" * 70)