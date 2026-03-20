"""Example: hello world plugin for the harness.

Drop this file in ~/.z/plugins/ or <project>/.z/plugins/ to auto-load.
Or add its path to "plugins" in ~/.z.json.

Demonstrates:
  - Registering a custom tool the LLM can call
  - Subscribing to lifecycle hooks
  - Reading per-plugin config
"""

import datetime


def register(api):
    """Entry point called by the plugin manager."""

    # Read any per-plugin config from ~/.z.json → plugin_config.hello_world
    cfg = api.get_config()
    greeting = cfg.get("greeting", "Hello")

    # Register a tool the LLM can invoke
    api.add_tool(
        name="hello_world",
        description=f"A demo plugin tool that greets someone. Says '{greeting}'.",
        params={
            "name": {"required": True, "description": "Who to greet"},
        },
        handler=lambda params: f"{greeting}, {params.get('name', 'world')}! "
                               f"(from hello_world plugin at {datetime.datetime.now():%H:%M:%S})",
        console_label="[green]Hello[/green]",
    )

    # Example hook: append a line to the system prompt
    api.on("system_prompt", lambda: (
        "The hello_world plugin is loaded. "
        "You can use the hello_world tool to greet people."
    ))
