"""Test plugin: quick_search — a lightweight serper wrapper with result tracking.

Demonstrates the full plugin lifecycle:
  - add_tool: registers a search tool with a focus parameter
  - get_config: reads per-plugin settings
  - on("system_prompt"): adds model guidance
  - on("pre_tool"): logs incoming tool calls
  - on("post_tool"): tracks serper usage and formats results
"""

_searches_today = 0


def register(api):
    cfg = api.get_config()
    default_focus = cfg.get("default_focus", "general")

    def quick_search(params):
        """Route a search query to serper, optionally scoping by focus area."""
        query = params.get("query", "")
        focus = params.get("focus", default_focus)

        if not query:
            return "Error: 'query' parameter is required"

        # Build a scoped query if focus is set
        scoped_query = f"{query} ({focus})" if focus and focus != "general" else query

        return (
            f"I'll search for: {scoped_query}\n"
            f"To complete this, call mcp_call_tool with server='serper', "
            f"tool='search', arguments='{{\"query\": \"{scoped_query}\"}}'"
        )

    api.add_tool(
        name="quick_search",
        description=(
            "Quickly search the web using Serper. Use this for fast lookups. "
            "The results come back raw — summarize them for the user."
        ),
        params={
            "query": {"required": True, "description": "The search query"},
            "focus": {
                "required": False,
                "description": "Optional focus area to scope results (e.g. 'python', 'finance', 'news')",
            },
        },
        handler=quick_search,
        console_label="[cyan]QuickSearch[/cyan]",
    )

    # Hook: tell the model how to use this
    api.on("system_prompt", lambda: (
        "The quick_search plugin is loaded. When you need fast web lookups, "
        "use quick_search first — it will tell you what serper MCP call to make. "
        "Always cite sources from the search results."
    ))

    # Hook: log all tool calls
    def on_pre_tool(tool_name, params):
        from harness.logger import get_logger
        get_logger("quick_search").info(">>> %s(%s)", tool_name, params)

    api.on("pre_tool", on_pre_tool)

    # Hook: track serper usage
    def on_post_tool(tool_name, params, result):
        global _searches_today
        if tool_name == "mcp_call_tool" and params.get("server") == "serper":
            _searches_today += 1
            from harness.logger import get_logger
            get_logger("quick_search").info(
                "serper query #%d: %s", _searches_today, params.get("arguments", "?")
            )

    api.on("post_tool", on_post_tool)
