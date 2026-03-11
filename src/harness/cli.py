#!/usr/bin/env python3
"""Entry point for the harness CLI."""

import os
import importlib.util

# Directly load the root harness.py module by file path
# This bypasses the package vs module naming conflict
current_dir = os.path.dirname(os.path.abspath(__file__))
harness_file = os.path.abspath(os.path.join(current_dir, "..", "..", "harness.py"))

spec = importlib.util.spec_from_file_location("root_harness", harness_file)
harness_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness_module)

main = harness_module.main


def run():
    """Run the harness CLI."""
    main()


if __name__ == "__main__":
    run()