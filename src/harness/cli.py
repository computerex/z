"""CLI entry point for the z command (pip-installed).

This is a thin wrapper that delegates to harness.main's main() function.
"""
from harness.main import main

def run():
    """Entry point for the 'z' console script."""
    return main() or 0
