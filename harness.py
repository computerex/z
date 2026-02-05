#!/usr/bin/env python3
"""Harness launcher - run from anywhere."""

import sys
import os

# Force unbuffered stdout/stderr for real-time streaming
sys.stdout.reconfigure(line_buffering=False)
sys.stderr.reconfigure(line_buffering=False)
os.environ['PYTHONUNBUFFERED'] = '1'

# Add the src directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.cli import main

if __name__ == "__main__":
    main()
