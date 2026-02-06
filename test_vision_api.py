"""Test script to verify Z.AI vision API works via Anthropic endpoint."""

import asyncio
import base64
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def test_vision():
    api_key = os.getenv('LLM_API_KEY')
    
    if not api_key:
        print('ERROR: LLM_API_KEY not set in .env or environment')
        return
    
    # Z.AI Anthropic endpoint (supports vision)
    anthropic_url = 'https://api.z.ai/api/anthropic/v1/messages'
    print(f'Using Anthropic endpoint: {anthropic_url}')
    print(f'API key: {api_key[:8]}...{api_key[-4:]}')
    print()
    
    # Create a small test image (1x1 red pixel PNG)
    test_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
    img_b64 = base64.b64encode(test_png).decode()
    
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }
    
    # Test with Anthropic format
    print('='*60)
    print('Test: glm-4.7 vision via Anthropic endpoint')
    print('='*60)
    payload = {
        'model': 'glm-4.7',
        'max_tokens': 100,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64}},
                {'type': 'text', 'text': 'What color is this 1x1 pixel image?'}
            ]
        }]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(anthropic_url, headers=headers, json=payload)
            print(f'Status: {resp.status_code}')
            data = resp.json()
            if 'content' in data:
                print(f'Response: {data["content"][0].get("text", "")}')
            else:
                print(f'Response: {resp.text[:500]}')
        except Exception as e:
            print(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(test_vision())
