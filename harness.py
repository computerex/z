#!/usr/bin/env python3
"""Compatibility redirect — real entry point is harness.main:main."""
import sys
import os

# Ensure src/ is on path so harness package is importable
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from harness.main import main

if __name__ == '__main__':
    sys.exit(main() or 0)
