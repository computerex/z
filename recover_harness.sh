#!/bin/bash
# HARNESS RECOVERY SCRIPT
# Run this if the harness is completely broken and won't start
#
# This script:
# 1. Restores harness code to the last git commit
# 2. Tests if it works
# 3. If still broken, offers to run safe mode

echo "============================================================"
echo "  HARNESS RECOVERY"
echo "============================================================"
echo

cd "$(dirname "$0")"

echo "[1/3] Restoring harness code to last git commit..."
git checkout HEAD -- harness.py src/harness/
if [ $? -ne 0 ]; then
    echo "ERROR: git checkout failed"
    exit 1
fi
echo "      Done."
echo

echo "[2/3] Testing if harness works..."
python3 harness.py --help >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "      Harness still broken after restore!"
    echo
    echo "[3/3] Starting safe mode for repair..."
    python3 safe_harness.py --fix
    exit
fi

echo "      Harness is working!"
echo
echo "Recovery complete. You can now run: python3 harness.py"