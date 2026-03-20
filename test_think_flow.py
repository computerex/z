#!/usr/bin/env python3
"""
Test to trace where </think> gets lost in the harness.
This simulates the message flow from user input to API call.
"""

import json
import sys
import os


# Simulate the harness message flow
class StreamingMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.provider_blocks = None

    def to_dict(self):
        return {"role": self.role, "content": self.content}


def test_message_flow():
    """Test the complete message flow."""
    print("=" * 60)
    print("TESTING: Where does </think> get lost?")
    print("=" * 60)

    # Step 1: User types message
    user_input = "</think> test"
    print(f"\n1. USER INPUT: {repr(user_input)}")

    # Step 2: Message is added to list (cline_agent.py:1286)
    messages = [
        StreamingMessage(role="system", content="You are a helpful assistant."),
        StreamingMessage(role="user", content="previous message"),
        StreamingMessage(role="assistant", content="previous response"),
        StreamingMessage(role="user", content=user_input),
    ]
    print(f"\n2. AFTER ADDING TO MESSAGES LIST:")
    print(f"   Total messages: {len(messages)}")
    for i, m in enumerate(messages):
        print(f"   {i}: [{m.role}] {repr(m.content[:50])}")

    # Step 3: Check last message
    last_msg = messages[-1]
    print(f"\n3. LAST MESSAGE CHECK:")
    print(f"   Role: {last_msg.role}")
    print(f"   Content: {repr(last_msg.content)}")
    print(f"   Contains </think>: {'</think>' in last_msg.content}")

    # Step 4: Convert to dict (streaming_client.py:1077)
    messages_dict = [m.to_dict() for m in messages]
    print(f"\n4. AFTER to_dict() CONVERSION:")
    print(f"   Total dicts: {len(messages_dict)}")
    for i, m in enumerate(messages_dict):
        print(f"   {i}: [{m['role']}] {repr(m['content'][:50])}")

    # Step 5: Check if </think> is in the dict
    last_dict = messages_dict[-1]
    print(f"\n5. LAST DICT CHECK:")
    print(f"   Role: {last_dict['role']}")
    print(f"   Content: {repr(last_dict['content'])}")
    print(f"   Contains </think>: {'</think>' in last_dict['content']}")

    # Step 6: JSON serialization (streaming_client.py:1082)
    payload_json = json.dumps(messages_dict, ensure_ascii=False)
    print(f"\n6. JSON PAYLOAD (first 300 chars):")
    print(f"   {payload_json[:300]}...")

    # Step 7: Check if </think> survives JSON
    if "</think>" in payload_json:
        print(f"\n   [PASS] </think> is in JSON payload")
    else:
        print(f"\n   [FAIL] </think> was lost in JSON serialization!")
        return False

    # Step 8: Parse back to verify
    parsed = json.loads(payload_json)
    last_parsed = parsed[-1]
    print(f"\n7. AFTER JSON PARSE BACK:")
    print(f"   Content: {repr(last_parsed['content'])}")
    print(f"   Contains </think>: {'</think>' in last_parsed['content']}")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if "</think>" in last_parsed["content"]:
        print("[PASS] </think> survives serialization!")
        print("The bug is likely in the smart context compaction or")
        print("in how the API handles the message.")
    else:
        print("[FAIL] </think> was lost during serialization!")
    print("=" * 60)

    return "</think>" in last_parsed["content"]


if __name__ == "__main__":
    success = test_message_flow()
    sys.exit(0 if success else 1)
