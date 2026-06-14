#!/usr/bin/env python3
"""Entry point for the z command."""

import sys
import os

# Add src directory to path to find the harness package
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add src to path
_src_dir = os.path.join(_project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Now import and run the main function from harness.py
from harness import main

if __name__ == '__main__':
    sys.exit(main() or 0)
