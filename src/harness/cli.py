"""CLI entry point for the z command (pip-installed).

This is a thin wrapper that delegates to harness.py's main() function.
"""
import sys
import os

# __file__ is .../src/harness/cli.py
# Project root is .../ (grandparent of src)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "harness_main", os.path.join(_project_root, "harness.py")
)
_harness_module = importlib.util.module_from_spec(_spec)
sys.modules["harness_main"] = _harness_module
_spec.loader.exec_module(_harness_module)


def run():
    """Entry point for the 'z' console script."""
    return _harness_module.main() or 0
