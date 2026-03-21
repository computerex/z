@echo off
:: HARNESS RECOVERY SCRIPT
:: Run this if the harness is completely broken and won't start
::
:: This script:
:: 1. Restores harness code to the last git commit
:: 2. Tests if it works
:: 3. If still broken, offers to run safe mode

echo ============================================================
echo   HARNESS RECOVERY
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Restoring harness code to last git commit...
git checkout HEAD -- harness.py src/harness/
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: git checkout failed
    pause
    exit /b 1
)
echo       Done.
echo.

echo [2/3] Testing if harness works...
python harness.py --help >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo       Harness still broken after restore!
    echo.
    echo [3/3] Starting safe mode for repair...
    python safe_harness.py --fix
    exit /b
)

echo       Harness is working!
echo.
echo Recovery complete. You can now run: python harness.py
echo.
pause