"""CLI subpackage — install wizard, provider management, sessions."""
from ..main import main


def run():
    """Entry point for the 'z' console script (backward compat)."""
    return main() or 0


# Individual modules importable as: from harness.cli.install import run_install
# from harness.cli.providers import load_providers
# from harness.cli.sessions import list_sessions
