#!/usr/bin/env python3
"""Console-script entry point for the harness CLI.

Registered in pyproject.toml as: z = "harness.cli:run"
"""

from harness._app import main


def run():
    """Run the harness CLI."""
    main()


if __name__ == "__main__":
    run()