#!/usr/bin/env python3
"""
HEADLESS TEST: Find where </think> gets lost in the harness.
This simulates the entire message flow without requiring user interaction.
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional, Tuple


# Simulate harness classes
class StreamingMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.provider_blocks = None

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


def estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    return len(text) // 4


class MockSmartContext:
    """Mock smart context that might be reordering messages."""

    def __init__(self):
        self.soft_target_ratio = 0.5
        self.budget_ratio = 0.85
        self.recent_window = 6
        self.max_guidance_prune_per_tick = 2
        self.max_guidance_prune_per_tick_boosted = 5
        self.guidance_prune_boost_threshold = 0.9
        self.node_metadata = {}

    def semantic_maintenance_tick(
        self, messages: List[Any], max_tokens: int, current_tokens: Optional[int] = None
    ) -> Tuple[List[Any], int, str]:
        """Mock maintenance - check if it reorders messages."""
        print(f"\n  [SMART_CONTEXT] semantic_maintenance_tick called")
        print(f"  [SMART_CONTEXT] Input message count: {len(messages)}")

        # Check if messages are in correct order
        for i, msg in enumerate(messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            role = msg.role if hasattr(msg, "role") else "unknown"
            print(f"  [SMART_CONTEXT] Input {i}: [{role}] {repr(content[:50])}")

        # Return messages unchanged (for this test)
        return messages, 0, ""

    def compact_context(
        self, messages: List[Any], max_tokens: int, current_tokens: Optional[int] = None
    ) -> Tuple[List[Any], int, str]:
        """Mock compaction."""
        print(f"\n  [SMART_CONTEXT] compact_context called")
        return messages, 0, ""


def simulate_harness_flow():
    """Simulate the complete harness message flow."""
    print("=" * 70)
    print("HEADLESS TEST: Simulating harness message flow")
    print("=" * 70)

    # Simulate cline_agent state
    messages: List[StreamingMessage] = [
        StreamingMessage(role="system", content="You are a helpful assistant."),
    ]

    smart_context = MockSmartContext()
    max_allowed = 128000

    # Simulate conversation
    conversation = [
        ("user", "hi repeat after me ok?"),
        ("assistant", "OK, I'm ready. Go ahead and say what you'd like me to repeat."),
        ("user", "</think>"),
    ]

    for turn, (role, content) in enumerate(conversation):
        print(f"\n{'=' * 70}")
        print(f"TURN {turn + 1}: {role} says: {repr(content)}")
        print(f"{'=' * 70}")

        # Step 1: Add message (cline_agent.py:1286)
        print(f"\n[STEP 1] Adding message to list...")
        messages.append(StreamingMessage(role=role, content=content))
        print(f"  Message count: {len(messages)}")

        # Step 2: Calculate tokens
        print(f"\n[STEP 2] Calculating tokens...")
        total_chars = sum(len(m.content) for m in messages)
        token_count = sum(estimate_tokens(m.content) for m in messages)
        print(f"  Total chars: {total_chars}")
        print(f"  Estimated tokens: {token_count}")

        # Step 3: Semantic maintenance (cline_agent.py:1421)
        print(f"\n[STEP 3] Running semantic maintenance...")
        messages, freed, report = smart_context.semantic_maintenance_tick(
            messages, max_allowed, current_tokens=token_count
        )
        print(f"  Messages after maintenance: {len(messages)}")

        # Step 4: Compact context if needed (cline_agent.py:1452)
        compact_threshold = int(max_allowed * 0.85)
        if token_count > compact_threshold:
            print(f"\n[STEP 4] Running context compaction...")
            messages, freed, report = smart_context.compact_context(
                messages, max_allowed, current_tokens=token_count
            )
            print(f"  Messages after compaction: {len(messages)}")
        else:
            print(f"\n[STEP 4] Skipping compaction (under threshold)")

        # Step 5: Prepare for API call
        print(f"\n[STEP 5] Preparing API payload...")
        print(f"  Final message list ({len(messages)} messages):")
        for i, m in enumerate(messages):
            print(f"    {i}: [{m.role}] {repr(m.content[:60])}")

        # Step 6: Convert to dict (streaming_client.py:1077)
        print(f"\n[STEP 6] Converting to dict...")
        messages_dict = [m.to_dict() for m in messages]
        print(f"  Dict count: {len(messages_dict)}")
        for i, m in enumerate(messages_dict):
            print(f"    {i}: [{m['role']}] {repr(m['content'][:60])}")

        # Step 7: Verify last user message
        print(f"\n[STEP 7] Verification...")
        last_user_msgs = [m for m in messages_dict if m["role"] == "user"]
        if last_user_msgs:
            last_user = last_user_msgs[-1]
            print(f"  Last user message: {repr(last_user['content'])}")
            # Only check for </think> if it was in the input
            if "</think>" in content:
                if "</think>" in last_user["content"]:
                    print(f"  [PASS] </think> IS preserved")
                else:
                    print(f"  [FAIL] </think> was LOST!")
                    return False
            else:
                print(f"  [INFO] No </think> in this turn's input")

        # Step 8: JSON serialization
        print(f"\n[STEP 8] JSON serialization...")
        payload_json = json.dumps(messages_dict, ensure_ascii=False)
        # Only check for </think> if it was in the input
        if "</think>" in content:
            if "</think>" in payload_json:
                print(f"  [PASS] </think> is in JSON payload")
            else:
                print(f"  [FAIL] </think> is missing from JSON!")
                return False
        else:
            print(f"  [INFO] No </think> check needed for this turn")

        print(f"\n{'=' * 70}")
        print(f"TURN {turn + 1} COMPLETE")
        print(f"{'=' * 70}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = simulate_harness_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
