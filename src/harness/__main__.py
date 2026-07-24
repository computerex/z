"""Main entry point for running the harness as a module (python -m harness)."""

from harness.cli import run

if __name__ == '__main__':
    import sys
    sys.exit(run() or 0)
