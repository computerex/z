"""Install wizard — interactive and headless provider setup."""
from pathlib import Path
from typing import Optional
import json

def run_install(
    api_url: str = None,
    api_key: str = None,
    model: str = None,
    global_config: bool = True,
):
    """Setup wizard for API configuration. Supports headless mode with CLI args.

    Args:
        api_url: API base URL (headless mode)
        api_key: API key (headless mode)
        model: Model name (headless mode)
        global_config: Deprecated; config is always saved to ~/.z.json
    """
    import json

    # Headless mode - all params provided
    if api_url and api_key:
        config_data = {
            "api_url": api_url.rstrip("/") + "/",
            "api_key": api_key,
            "model": model or "glm-4.7",
        }

        config_path = Path.home() / ".z.json"

        config_path.write_text(json.dumps(config_data, indent=2))
        print(f"Configuration saved to: {config_path}")
        print(f"  URL:   {api_url}")
        print(f"  Model: {config_data['model']}")
        print(f"  Key:   {api_key[:10]}...")
        return

    # Interactive mode
    con = Console()
    con.print()
    con.print(
        Panel(
            "[bold]Welcome to Harness[/bold]\n\n[dim]Let's configure your LLM provider.[/dim]",
            border_style="bright_blue",
            padding=(1, 3),
            width=50,
        )
    )
    con.print()
    con.print("  Select your LLM provider:\n")
    con.print("  [cyan][1][/cyan] Z.AI Coding Plan [dim](recommended)[/dim]")
    con.print("  [cyan][2][/cyan] Z.AI Standard API")
    con.print("  [cyan][3][/cyan] MiniMax")
    con.print("  [cyan][4][/cyan] Amazon Bedrock")
    con.print("  [cyan][5][/cyan] Together AI")
    con.print("  [cyan][6][/cyan] Anthropic")
    con.print("  [cyan][7][/cyan] OpenRouter")
    con.print("  [cyan][8][/cyan] OpenAI")
    con.print("  [cyan][9][/cyan] Groq")
    con.print("  [cyan][10][/cyan] DeepSeek")
    con.print("  [cyan][11][/cyan] Mistral AI")
    con.print("  [cyan][12][/cyan] Cohere")
    con.print("  [cyan][13][/cyan] Fireworks AI")
    con.print("  [cyan][14][/cyan] Perplexity")
    con.print("  [cyan][15][/cyan] AI21")
    con.print("  [cyan][16][/cyan] xAI (Grok)")
    con.print("  [cyan][17][/cyan] Google Gemini")
    con.print("  [cyan][18][/cyan] Cerebras")
    con.print("  [cyan][19][/cyan] Databricks")
    con.print("  [cyan][20][/cyan] Replicate")
    con.print("  [cyan][21][/cyan] Anyscale")
    con.print("  [cyan][22][/cyan] Ollama Cloud")
    con.print("  [cyan][23][/cyan] OpenAI Subscription (OAuth)")
    con.print("  [cyan][24][/cyan] GitHub Copilot (OAuth)")
    con.print("  [cyan][25][/cyan] Custom OpenAI-compatible API")
    con.print("  [cyan][26][/cyan] Local Ollama")
    con.print()

    while True:
        choice = input("Enter choice [1-26]: ").strip()
        if choice == "1":
            base_url = "https://api.z.ai/api/coding/paas/v4/"
            provider = "Z.AI Coding"
            default_model = "glm-4.7"
            break
        elif choice == "2":
            base_url = "https://api.z.ai/api/paas/v4/"
            provider = "Z.AI Standard"
            default_model = "glm-4.7"
            break
        elif choice == "3":
            base_url = "https://api.minimax.io/v1/"
            provider = "MiniMax"
            default_model = "MiniMax-M2.1"
            break
        elif choice == "4":
            base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
            provider = "Amazon Bedrock"
            default_model = "qwen.qwen3-32b-v1:0"
            break
        elif choice == "5":
            base_url = "https://api.together.xyz/v1/"
            provider = "Together AI"
            default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            break
        elif choice == "6":
            base_url = "https://api.anthropic.com/v1/"
            provider = "Anthropic"
            default_model = "claude-3-5-sonnet-latest"
            break
        elif choice == "7":
            base_url = "https://openrouter.ai/api/v1/"
            provider = "OpenRouter"
            default_model = "anthropic/claude-3.5-sonnet"
            break
        elif choice == "8":
            base_url = "https://api.openai.com/v1/"
            provider = "OpenAI"
            default_model = "gpt-4o"
            break
        elif choice == "9":
            base_url = "https://api.groq.com/openai/v1/"
            provider = "Groq"
            default_model = "llama-3.3-70b-versatile"
            break
        elif choice == "10":
            base_url = "https://api.deepseek.com/v1/"
            provider = "DeepSeek"
            default_model = "deepseek-chat"
            break
        elif choice == "11":
            base_url = "https://api.mistral.ai/v1/"
            provider = "Mistral AI"
            default_model = "mistral-large-latest"
            break
        elif choice == "12":
            base_url = "https://api.cohere.ai/v1/"
            provider = "Cohere"
            default_model = "command-r-plus"
            break
        elif choice == "13":
            base_url = "https://api.fireworks.ai/inference/v1/"
            provider = "Fireworks AI"
            default_model = "accounts/fireworks/models/llama-v3p1-70b-instruct"
            break
        elif choice == "14":
            base_url = "https://api.perplexity.ai/"
            provider = "Perplexity"
            default_model = "llama-3.1-sonar-large-128k-online"
            break
        elif choice == "15":
            base_url = "https://api.ai21.com/studio/v1/"
            provider = "AI21"
            default_model = "jamba-1.5-large"
            break
        elif choice == "16":
            base_url = "https://api.x.ai/v1/"
            provider = "xAI (Grok)"
            default_model = "grok-2-latest"
            break
        elif choice == "17":
            base_url = "https://generativelanguage.googleapis.com/v1beta/"
            provider = "Google Gemini"
            default_model = "gemini-1.5-pro-latest"
            break
        elif choice == "18":
            base_url = "https://api.cerebras.ai/v1/"
            provider = "Cerebras"
            default_model = "llama3.1-70b"
            break
        elif choice == "19":
            base_url = input(
                "Databricks workspace URL (e.g., https://my-workspace.cloud.databricks.com/serving-endpoints/): "
            ).strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Databricks"
            default_model = "databricks-meta-llama-3-1-70b-instruct"
            break
        elif choice == "20":
            base_url = "https://api.replicate.com/v1/"
            provider = "Replicate"
            default_model = "meta/meta-llama-3-70b-instruct"
            break
        elif choice == "21":
            base_url = "https://api.endpoints.anyscale.com/v1/"
            provider = "Anyscale"
            default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
            break
        elif choice == "22":
            base_url = "https://ollama.com/v1/"
            provider = "Ollama Cloud"
            default_model = "llama3.1"
            break
        elif choice == "23":
            base_url = "https://api.openai.com/v1/"
            provider = "OpenAI Subscription (OAuth)"
            default_model = "gpt-4o"
            break
        elif choice == "24":
            base_url = "https://api.githubcopilot.com/"
            provider = "GitHub Copilot (OAuth)"
            default_model = "gpt-4o-copilot"
            break
        elif choice == "25":
            base_url = input("Enter API base URL: ").strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Custom"
            default_model = input("Enter default model name: ").strip() or "gpt-4"
            break
        elif choice == "26":
            base_url = "http://localhost:11434/v1"
            provider = "Local Ollama"
            default_model = ""
            # Try to detect Ollama and fetch available models
            try:
                import httpx
                r = httpx.get("http://localhost:11434/v1/models", timeout=5)
                if r.status_code == 200:
                    models = [m["id"] for m in r.json().get("data", [])]
                    if models:
                        con.print(f"\n  [dim]Found {len(models)} model(s):[/dim]")
                        for i, m in enumerate(models, 1):
                            con.print(f"    [{i}] {m}")
                        model_input = input(f"\n  Select model [1-{len(models)}]: ").strip()
                        if model_input.isdigit() and 1 <= int(model_input) <= len(models):
                            default_model = models[int(model_input) - 1]
                        else:
                            default_model = model_input or ""
                else:
                    con.print("  [yellow]Could not reach Ollama. Check that it is running.[/yellow]")
            except Exception:
                con.print("  [yellow]Could not reach Ollama. Check that it is running.[/yellow]")
            if not default_model:
                default_model = input("  Model name: ").strip()
            break
        else:
            print("Please enter 1-26.")

    con.print(f"\n  [green]\u2713[/green] Using [bold]{provider}[/bold]")
    con.print(f"    [dim]{base_url}[/dim]\n")

    # Check if OAuth provider
    is_oauth = "(OAuth)" in provider

    if is_oauth:
        # OAuth flow
        con.print("  [dim]This provider uses OAuth authentication.[/dim]")

        # Import OAuth manager
        try:
            from .providers import get_oauth_manager

            oauth_manager = get_oauth_manager()

            # Map provider name to OAuth provider ID
            if "OpenAI" in provider:
                oauth_provider_id = "openai"
            else:
                oauth_provider_id = "github-copilot"

            # For OpenAI, let user choose method
            oauth_method = "browser"
            enterprise_url = None
            if "OpenAI" in provider:
                con.print("\n  Select OAuth method:")
                con.print("  [1] Browser-based (opens browser for authorization)")
                con.print("  [2] Device code (headless, enter code manually)")
                method_choice = input("\n  Enter choice [1/2]: ").strip()
                oauth_method = "device" if method_choice == "2" else "browser"
            elif "GitHub Copilot" in provider:
                # GitHub Copilot only supports device code flow
                con.print("\n  GitHub Copilot uses device code authentication.")

                # Ask about GitHub Enterprise
                is_enterprise = (
                    input("  Is this GitHub Enterprise? [y/N]: ").strip().lower()
                )
                if is_enterprise in ("y", "yes"):
                    enterprise_url = input(
                        "  Enter GitHub Enterprise domain (e.g., company.ghe.com): "
                    ).strip()

            con.print("\n  Opening browser for authentication...\n")

            # Trigger OAuth flow with selected method
            token = oauth_manager.authenticate(
                oauth_provider_id,
                method=oauth_method,
                timeout=300,
                enterprise_url=enterprise_url,
            )
            if token:
                api_key = f"oauth:{token.access_token}"
                con.print(f"  [green]✓[/green] OAuth authentication successful!\n")
            else:
                con.print("  [red]✗[/red] OAuth authentication failed.\n")
                return
        except Exception as e:
            con.print(f"  [red]✗[/red] OAuth error: {e}\n")
            return
    else:
        # API Key flow
        api_key = ""
        while not api_key:
            api_key = input("API Key: ").strip()
            if not api_key:
                print("API key is required.")

    # Model
    if is_oauth:
        from .providers import get_codex_models
        from .providers import get_copilot_models

        if "GitHub Copilot" in provider:
            print(
                "  [dim]Note: GitHub Copilot OAuth tokens access Copilot models directly.[/dim]"
            )
            copilot_models = get_copilot_models()
            print(f"\n  Available Copilot models:")
            for i, m in enumerate(copilot_models, 1):
                marker = "●" if m == default_model else " "
                print(f"    {marker} [{i}] {m}")

            print(
                f"\n  Select model [1-{len(copilot_models)}] or enter name (default: {default_model}): ",
                end="",
                flush=True,
            )
            model_choice = input().strip()
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(copilot_models):
                model = copilot_models[int(model_choice) - 1]
            else:
                model = model_choice or default_model
        else:
            print(
                "  [dim]Note: OAuth tokens access ChatGPT Codex models directly.[/dim]"
            )

            # Show available Codex models (hardcoded for instant display)
            codex_models = get_codex_models()
            print(f"\n  Available Codex models:")
            for i, m in enumerate(codex_models, 1):
                marker = "●" if m == default_model else " "
                print(f"    {marker} [{i}] {m}")

            print(
                f"\n  Select model [1-{len(codex_models)}] or enter name (default: {default_model}): ",
                end="",
                flush=True,
            )
            model_choice = input().strip()
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(codex_models):
                model = codex_models[int(model_choice) - 1]
            else:
                model = model_choice or default_model
    else:
        model_input = input(f"\nModel name (default: {default_model}): ").strip()
        model = model_input or default_model

    # Build config
    config_data = {
        "api_url": base_url,
        "api_key": api_key,
        "model": model,
    }

    config_dir = Path.home()
    config_path = config_dir / ".z.json"
    location = "global"

    # Create directory if needed
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config_path.write_text(json.dumps(config_data, indent=2))

    tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
    tbl.add_column("label", style="dim", width=10, justify="right")
    tbl.add_column("value")
    tbl.add_row("Saved to", str(config_path))
    tbl.add_row("Location", location)
    tbl.add_row("Provider", provider)
    tbl.add_row("Model", model)
    tbl.add_row("Key", api_key[:10] + "...")
    con.print()
    con.print(
        Panel(
            tbl,
            title="[bold green] Setup Complete [/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    con.print("\n  [dim]Run [white]z[/white] to start.[/dim]\n")

