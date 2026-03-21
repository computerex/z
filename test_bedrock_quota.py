"""Investigate Bedrock daily token quotas and find workarounds.

Key findings so far:
- Error: "Too many tokens per day, please wait before trying again."
- This is a DAILY quota, not a rate limit
- No Retry-After header provided
- Model: us.anthropic.claude-opus-4-6-v1
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Load config
config_path = Path.home() / ".z.json"
config = json.loads(config_path.read_text())
bedrock_cfg = config.get("providers", {}).get("bedrock", {})

API_KEY = bedrock_cfg.get("api_key", "")
REGION = "us-east-1"

# Setup boto3
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

mgmt = session.client("bedrock", region_name=REGION)


def try_request(model_id, label=""):
    """Try a minimal converse request, return success/error."""
    try:
        resp = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
            system=[{"text": "Be brief."}],
            inferenceConfig={"maxTokens": 5, "temperature": 0.1},
        )
        text = ""
        for b in resp.get("output", {}).get("message", {}).get("content", []):
            text += b.get("text", "")
        usage = resp.get("usage", {})
        return True, f"OK: '{text}' ({usage.get('inputTokens',0)}in/{usage.get('outputTokens',0)}out)"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        return False, f"{code}: {msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_service_quotas():
    """Try to read Bedrock service quotas from AWS Service Quotas API."""
    print("\n=== Service Quotas ===")
    try:
        sq = session.client("service-quotas", region_name=REGION)
        
        # List all Bedrock quotas
        paginator = sq.get_paginator("list_service_quotas")
        quotas = []
        for page in paginator.paginate(ServiceCode="bedrock"):
            quotas.extend(page.get("Quotas", []))
        
        print(f"  Found {len(quotas)} Bedrock quotas:")
        for q in quotas:
            name = q.get("QuotaName", "")
            value = q.get("Value", "N/A")
            unit = q.get("Unit", "")
            # Filter for relevant ones (Claude, Opus, token, daily)
            name_lower = name.lower()
            if any(kw in name_lower for kw in ("claude", "opus", "anthropic", "token", "daily")):
                adjustable = "adjustable" if q.get("Adjustable") else "fixed"
                print(f"    {name}: {value} {unit} ({adjustable})")
                print(f"      Code: {q.get('QuotaCode', 'N/A')}")
        
        # Also search for any quota with "token" in the name
        print("\n  Token-related quotas:")
        for q in quotas:
            name = q.get("QuotaName", "")
            if "token" in name.lower():
                value = q.get("Value", "N/A")
                unit = q.get("Unit", "")
                print(f"    {name}: {value} {unit}")
                
    except Exception as e:
        print(f"  Error reading quotas: {e}")


def check_cloudwatch_metrics():
    """Try to read Bedrock usage metrics from CloudWatch."""
    print("\n=== CloudWatch Bedrock Metrics ===")
    try:
        cw = session.client("cloudwatch", region_name=REGION)
        
        # List available Bedrock metrics
        resp = cw.list_metrics(Namespace="AWS/Bedrock")
        metrics = resp.get("Metrics", [])
        
        print(f"  Found {len(metrics)} Bedrock metrics:")
        seen = set()
        for m in metrics:
            name = m.get("MetricName", "")
            if name not in seen:
                seen.add(name)
                dims = {d["Name"]: d["Value"] for d in m.get("Dimensions", [])}
                print(f"    {name} {dims}")
                
    except Exception as e:
        print(f"  Error: {e}")


def test_alternative_models():
    """Test different model IDs to see if any have separate/larger quotas."""
    print("\n=== Testing Alternative Model IDs ===")
    
    models = [
        ("us.anthropic.claude-opus-4-6-v1", "US inference profile (current)"),
        ("global.anthropic.claude-opus-4-6-v1", "Global inference profile"),
        ("anthropic.claude-opus-4-6-v1", "Base model ID"),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", "Opus 4.5 US"),
        ("global.anthropic.claude-opus-4-5-20251101-v1:0", "Opus 4.5 Global"),
        ("us.anthropic.claude-opus-4-20250514-v1:0", "Opus 4.0 US"),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "Opus 4.1 US"),
        ("us.anthropic.claude-sonnet-4-6", "Sonnet 4.6 US"),
        ("global.anthropic.claude-sonnet-4-6", "Sonnet 4.6 Global"),
    ]
    
    for model_id, desc in models:
        ok, detail = try_request(model_id, desc)
        status = "✓" if ok else "✗"
        print(f"  {status} {desc}")
        print(f"    Model: {model_id}")
        print(f"    Result: {detail}")
        print()
        
        # Small delay between tests
        time.sleep(1)


def check_inference_profiles_detail():
    """Get detailed info about available inference profiles."""
    print("\n=== Inference Profile Details ===")
    try:
        resp = mgmt.list_inference_profiles()
        profiles = resp.get("inferenceProfileSummaries", [])
        
        for p in profiles:
            pid = p.get("inferenceProfileId", "")
            # Only show Claude profiles
            if "claude" not in pid.lower():
                continue
            if "opus" not in pid.lower() and "sonnet" not in pid.lower():
                continue
                
            print(f"  {pid}")
            print(f"    Status: {p.get('status')}")
            print(f"    Type: {p.get('type')}")
            print(f"    Description: {p.get('description', 'N/A')}")
            
            # Check for model access
            models = p.get("models", [])
            if models:
                for m in models:
                    print(f"    Underlying model: {m.get('modelArn', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print(f"Bedrock Quota Investigation - {datetime.now().isoformat()}")
    print("=" * 70)
    
    # 1. Check service quotas
    check_service_quotas()
    
    # 2. Check CloudWatch metrics  
    check_cloudwatch_metrics()
    
    # 3. Get inference profile details
    check_inference_profiles_detail()
    
    # 4. Test alternative models (some may have separate quotas)
    test_alternative_models()
    
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)