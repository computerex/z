#!/usr/bin/env python3
"""Streaming harness entry point - true token-by-token streaming."""

import sys
import os

# Force unbuffered output BEFORE any imports
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(write_through=True)

import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config
from harness.streaming_agent import StreamingAgent
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from rich.console import Console
from rich.panel import Panel


async def run(user_input: str, workspace: str):
    """Run the streaming agent."""
    os.chdir(workspace)
    
    # Load config from harness directory's .env
    harness_dir = Path(__file__).parent
    config = Config.from_env(harness_dir / ".env")
    config.validate()
    
    reset_global_tracker()
    
    agent = StreamingAgent(config)
    
    result = await agent.run(user_input)
    
    console = Console()
    
    # Show final response
    if result:
        console.print()
        console.print(Panel(result, title="Response", border_style="green"))
    
    # Show cost
    cost = get_global_tracker().get_summary()
    console.print(Panel(
        f"API Calls: {cost.total_calls}\n"
        f"Tokens: {cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out\n"
        f"Cost: ${cost.total_cost:.4f}",
        title="Cost",
        border_style="blue"
    ))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Harness")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace directory")
    args = parser.parse_args()
    
    # Resolve workspace
    if args.workspace == ".":
        workspace = os.getcwd()
    else:
        workspace = os.path.abspath(args.workspace)
    
    # Get input from stdin or prompt
    if not sys.stdin.isatty():
        user_input = sys.stdin.read().strip()
    else:
        print("Enter your request (Ctrl+D to submit):")
        user_input = sys.stdin.read().strip()
    
    if not user_input:
        print("No input provided.", file=sys.stderr)
        sys.exit(1)
    
    asyncio.run(run(user_input, workspace))


if __name__ == "__main__":
    main()
