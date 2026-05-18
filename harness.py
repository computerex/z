#!/usr/bin/env python3
"""Dev shim — the real code lives in src/harness/_app.py.

This file exists so `python harness.py` still works from a fresh clone
before/without `pip install`. Installed users get the `z` console script
via [project.scripts] in pyproject.toml.

We deliberately avoid `from harness._app import main` here because this
file (harness.py) lives next to the harness/ package directory and would
shadow it on sys.path. Instead we ensure src/ is on sys.path FIRST so the
real package wins, then import normally.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")

# Put src/ ahead of the script's own directory so `import harness` resolves
# to the package, not to this shim file.
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

# Drop the script's directory from sys.path to be sure this shim can't
# shadow the package via `import harness`.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from harness._app import main  # noqa: E402

if __name__ == "__main__":
    main()