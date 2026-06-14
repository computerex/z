#!/usr/bin/env python3
"""Main entry point for running the harness as a module."""

import sys
import os

# Ensure the project root is on the path for importing harness.py
# __file__ is .../src/harness/__main__.py
# Project root is .../ (grandparent of src)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import and run the main function from harness.py (at project root)
# This works because __main__.py runs as a module, so imports resolve correctly
import importlib.util
spec = importlib.util.spec_from_file_location("harness_main", os.path.join(_project_root, "harness.py"))
harness_module = importlib.util.module_from_spec(spec)
sys.modules["harness_main"] = harness_module
spec.loader.exec_module(harness_module)

def main():
    """Entry point that calls the main function from harness.py."""
    return harness_module.main()

if __name__ == '__main__':
    sys.exit(main() or 0)
