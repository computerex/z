"""Test script to debug GitHub Copilot reasoning events."""

import sys

sys.path.insert(0, "src")

import asyncio
from harness.oauth import get_oauth_manager
from harness.copilot_oauth_client import CopilotOAuthClient, CopilotMessage


async def test_copilot_reasoning():
    """Test Copilot API and log all event types."""

    # Get the saved GitHub Copilot token
    oauth_manager = get_oauth_manager()
    token = oauth_manager.get_token("github-copilot")

    if not token:
        print("ERROR: No GitHub Copilot token found!")
        print("Please authenticate first with: /providers setup github-copilot")
        return

    print(f"Using token: {token.access_token[:20]}...")
    print(f"Provider: {token.provider}")
    print()

    # Create client
    client = CopilotOAuthClient(
        oauth_token=token,
        model="claude-opus-4.6",
        temperature=0.7,
        max_tokens=4096,
        timeout=60.0,
    )

    # Track all event types
    event_types = set()

    async with client:
        messages = [
            CopilotMessage(role="user", content="Explain quicksort vs mergesort")
        ]

        print("Sending request to Copilot API...")
        print()

        response = await client.chat_stream(
            messages=messages,
            on_content=lambda chunk: print(f"[CONTENT] {chunk[:50]}...", end="\r"),
            on_reasoning=lambda chunk: print(f"[REASONING] {chunk[:50]}..."),
        )

        print()
        print()
        print("=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(
            f"Event types received: {sorted(event_types) if event_types else 'None tracked'}"
        )
        print(f"Content length: {len(response.content)} chars")
        print(
            f"Thinking length: {len(response.thinking) if response.thinking else 0} chars"
        )
        print(f"Has thinking: {response.thinking is not None}")

        if response.thinking:
            print()
            print("THINKING PREVIEW:")
            print(response.thinking[:500])


if __name__ == "__main__":
    asyncio.run(test_copilot_reasoning())
