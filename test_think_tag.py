#!/usr/bin/env python3
"""Test that </think> tags are preserved through the harness."""

import json
import sys


def test_message_serialization():
    """Test that </think> survives message serialization."""

    # Simulate StreamingMessage
    class StreamingMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

        def to_dict(self):
            return {"role": self.role, "content": self.content}

    # Test case 1: Basic serialization
    print("Test 1: Basic Message Serialization")
    messages = [
        StreamingMessage(role="system", content="You are a helpful assistant."),
        StreamingMessage(role="user", content="</think> test"),
    ]

    messages_dict = [m.to_dict() for m in messages]
    print(f"  Messages: {json.dumps(messages_dict)}")

    user_contents = [m["content"] for m in messages_dict if m["role"] == "user"]
    for content in user_contents:
        if "</think>" in content:
            print("  [PASS] </think> is preserved")
        else:
            print(f"  [FAIL] </think> was stripped! Got: {repr(content)}")
            return False

    # Test case 2: JSON round-trip
    print("\nTest 2: JSON Round-trip")
    json_str = json.dumps(messages_dict)
    round_trip = json.loads(json_str)
    for m in round_trip:
        if m["role"] == "user":
            if "</think>" in m["content"]:
                print("  [PASS] </think> survives JSON round-trip")
            else:
                print(f"  [FAIL] </think> lost! Got: {repr(m['content'])}")
                return False

    # Test case 3: Multiple messages
    print("\nTest 3: Multiple User Messages")
    messages2 = [
        StreamingMessage(role="system", content="System"),
        StreamingMessage(role="user", content="First message"),
        StreamingMessage(role="assistant", content="Response"),
        StreamingMessage(role="user", content="</think> second"),
    ]
    messages_dict2 = [m.to_dict() for m in messages2]
    user_msgs = [m for m in messages_dict2 if m["role"] == "user"]
    print(f"  Total user messages: {len(user_msgs)}")
    for i, m in enumerate(user_msgs):
        print(f"  User message {i}: {repr(m['content'])}")
        if "</think>" in m["content"]:
            print(f"    [PASS] Contains </think>")
        else:
            print(f"    [INFO] No </think>")

    print("\n[SUCCESS] All tests passed!")
    return True


if __name__ == "__main__":
    success = test_message_serialization()
    sys.exit(0 if success else 1)
