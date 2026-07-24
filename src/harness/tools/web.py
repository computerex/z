"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from urllib.parse import urlparse

async def web_search(self, params: Dict[str, str]) -> str:
    """Search the web using Z.AI's built-in web search via chat completions."""
    import httpx
    from urllib.parse import urlparse

    query = params.get("query", "")
    if not query:
        return "Error: search query is required"

    count = int(params.get("count", "5"))
    count = max(1, min(10, count))  # Clamp to 1-10

    # Use chat completion with web_search tool enabled
    parsed = urlparse(self.config.api_url)
    search_url = f"{parsed.scheme}://{parsed.netloc}/api/coding/paas/v4/chat/completions"

    def _build_payload(q: str, attempt: int) -> dict:
        msg = f"Search the web for: {q}"
        # Retry hint when the model returns unresolved function metadata.
        if attempt > 0:
            msg = (
                f"Search the web for: {q}\n"
                "Return final search results and summary text directly. "
                "Do not return tool/function call metadata."
            )
        return {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": msg}],
            "temperature": 0.4 if attempt > 0 else 0.7,
            "max_tokens": 2048,
            "stream": False,
            "tools": [{
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search-prime",
                    "search_result": True,
                    "count": str(count),
                }
            }]
        }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.config.api_key}",
        "Accept-Language": "en-US,en"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            data = {}
            results = []
            content = ""
            raw_fc_content = ""
            for attempt in range(2):
                payload = _build_payload(query, attempt)
                response = await http_client.post(search_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                results = data.get("web_search", [])
                content = ""
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")

                # Some responses occasionally return unresolved function-call JSON
                # like {"function":"google_search","arguments":"..."} in content.
                unresolved_fc = False
                raw_fc_content = ""
                if isinstance(content, str):
                    ct = content.strip()
                    if ct.startswith("{") and ct.endswith("}"):
                        try:
                            parsed = json.loads(ct)
                            if isinstance(parsed, dict) and "function" in parsed and "arguments" in parsed:
                                unresolved_fc = True
                                raw_fc_content = ct
                        except Exception:
                            pass

                if unresolved_fc and not results and attempt == 0:
                    # Transient backend behavior: retry once with clearer instruction.
                    continue
                break

            if not results and not content:
                return f"No results found for: {query}"
            if not results and raw_fc_content:
                return (
                    "Search backend returned unresolved function-call metadata instead of final results. "
                    "Please retry once.\n\n"
                    f"Raw content: {raw_fc_content[:300]}"
                )

            # Format results
            output = [f"Web Search Results for: {query}\n"]

            if results:
                output.append(f"Found {len(results)} sources:\n")
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    link = r.get("link", "")
                    media = r.get("media", "")
                    date = r.get("publish_date", "")
                    snippet = r.get("content", "")[:200]

                    output.append(f"[{i}] {title}")
                    if media:
                        output.append(f"    Source: {media}")
                    if date:
                        output.append(f"    Date: {date}")
                    if snippet:
                        output.append(f"    {snippet}...")
                    if link:
                        output.append(f"    URL: {link}")
                    output.append("")

            if content:
                output.append(f"\nSummary:\n{content}")

            result_text = "\n".join(output)

            # Add to context
            ctx_id = self.context.add("web_search", query, result_text)
            return f"[Context ID: {ctx_id}]\n\n{result_text}"

    except httpx.HTTPStatusError as e:
        return f"Error calling search API: {e.response.status_code} - {e.response.text[:200]}"
    except httpx.TimeoutException:
        return f"Error: Search request timed out after 120 seconds"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error searching web: {type(e).__name__}: {e}\n{tb}"

