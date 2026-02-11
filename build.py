#!/usr/bin/env python3
"""
Build script to package the harness into standalone executables.

Usage:
    python build.py              # Build for current platform
    python build.py --clean      # Clean build artifacts first
    python build.py --onedir     # Build as directory (faster startup, larger size)
    python build.py --linux      # Build Linux executable via Docker

Requirements:
    pip install pyinstaller
    Docker (for --linux cross-compilation)

Note: Native cross-compilation is not supported. Use --linux flag to build
Linux binaries via Docker from any platform.
    - Windows: produces harness.exe
    - Linux: produces harness
    - macOS: produces harness
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Build configuration
APP_NAME = "harness"
ENTRY_POINT = "harness.py"
ICON_FILE = None  # Set to path of .ico/.icns file if desired

# Additional data files to include
DATA_FILES = [
    # (source, destination_folder)
    # Example: ("config.json", "."),
]

# Hidden imports that PyInstaller might miss
HIDDEN_IMPORTS = [
    "harness",
    "harness.cline_agent",
    "harness.config",
    "harness.context_management",
    "harness.cost_tracker",
    "harness.interrupt",
    "harness.prompts",
    "harness.streaming_agent",
    "harness.streaming_client",
    "harness.tools",
    "harness.tools.registry",
    "harness.tools.file_tools",
    "harness.tools.shell_tools",
    "harness.tools.search_tools",
    "harness.todo_manager",
    "harness.smart_context",
    "httpx",
    "httpx._transports",
    "httpx._transports.default",
    "pydantic",
    "pydantic.main",
    "pydantic._internal",
    "pydantic_core",
    "tiktoken",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
    "rich",
    "rich.console",
    "rich.panel",
    "rich.markup",
    "prompt_toolkit",
    "prompt_toolkit.key_binding",
    "prompt_toolkit.keys",
    "prompt_toolkit.history",
    "prompt_toolkit.auto_suggest",
    "asyncio",
    "json",
    "re",
    "pathlib",
    "encodings.utf_8",
    "encodings.cp1252",
    "encodings.ascii",
    "aiofiles",
    "anyio",
    "sniffio",
    "h11",
    "httpcore",
    "certifi",
]

# Packages to exclude (reduce size)
EXCLUDES = [
    "matplotlib",
    "PIL",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
    "pytest",
    "pytest_asyncio",
    # Heavy ML packages - uncomment if not needed
    # "torch",
    # "transformers", 
    # "sentence_transformers",
    # "numpy",
]


def get_platform_info():
    """Get platform-specific build info."""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "name": "windows",
            "exe_name": f"{APP_NAME}.exe",
            "separator": ";",
        }
    elif system == "darwin":
        return {
            "name": "macos",
            "exe_name": APP_NAME,
            "separator": ":",
        }
    else:  # Linux and others
        return {
            "name": "linux",
            "exe_name": APP_NAME,
            "separator": ":",
        }


def clean_build_artifacts(root: Path):
    """Remove previous build artifacts."""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = [f"{APP_NAME}.spec"]
    
    for dir_name in dirs_to_clean:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"Removing {dir_path}")
            shutil.rmtree(dir_path)
    
    for file_name in files_to_clean:
        file_path = root / file_name
        if file_path.exists():
            print(f"Removing {file_path}")
            file_path.unlink()


def build_executable(root: Path, onefile: bool = True, debug: bool = False):
    """Build the executable using PyInstaller."""
    plat = get_platform_info()
    
    print(f"\n{'='*60}")
    print(f"Building {APP_NAME} for {plat['name']}")
    print(f"{'='*60}\n")
    
    # Base PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",  # Replace output without asking
        "--clean",      # Clean cache before building
    ]
    
    # One file or one directory
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # Console app (not windowed)
    cmd.append("--console")
    
    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])
    
    # Add excludes
    for exc in EXCLUDES:
        cmd.extend(["--exclude-module", exc])
    
    # Add data files
    for src, dst in DATA_FILES:
        cmd.extend(["--add-data", f"{src}{plat['separator']}{dst}"])
    
    # Add src directory as package
    cmd.extend(["--add-data", f"src{plat['separator']}src"])
    
    # Icon file if specified
    if ICON_FILE and Path(ICON_FILE).exists():
        cmd.extend(["--icon", ICON_FILE])
    
    # Debug mode
    if debug:
        cmd.append("--debug=all")
    
    # Optimize
    if not debug:
        # Note: --strip can cause issues on Windows, so we skip it there
        if platform.system() != "Windows":
            cmd.extend(["--strip"])  # Strip symbols (Linux/macOS only)
    
    # Entry point
    cmd.append(ENTRY_POINT)
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=root)
    
    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}")
        return False
    
    # Report success
    if onefile:
        exe_path = root / "dist" / plat["exe_name"]
    else:
        exe_path = root / "dist" / APP_NAME / plat["exe_name"]
    
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Build successful!")
        print(f"Executable: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\nWarning: Expected executable not found at {exe_path}")
        return False


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def build_linux_via_docker(root: Path) -> bool:
    """Build Linux executable using Docker."""
    print(f"\n{'='*60}")
    print("Building Linux executable via Docker")
    print(f"{'='*60}\n")
    
    # Check if Docker is available
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Docker is not available. Please install Docker.")
            return False
    except FileNotFoundError:
        print("Docker command not found. Please install Docker.")
        return False
    
    image_name = "harness-builder"
    container_name = "harness-build-temp"
    
    # Build the Docker image
    print("Building Docker image...")
    result = subprocess.run(
        ["docker", "build", "-f", "Dockerfile.build", "-t", image_name, "."],
        cwd=root,
    )
    if result.returncode != 0:
        print("Failed to build Docker image.")
        return False
    
    # Create dist directory if it doesn't exist
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    # Run the container to build
    print("\nBuilding executable in container...")
    
    # Remove any existing container with the same name
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )
    
    # Run the build
    result = subprocess.run(
        [
            "docker", "run",
            "--name", container_name,
            image_name,
        ],
        cwd=root,
    )
    
    if result.returncode != 0:
        print("Build failed in container.")
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        return False
    
    # Copy the built executable out
    print("\nCopying executable from container...")
    linux_exe = dist_dir / "harness-linux"
    result = subprocess.run(
        ["docker", "cp", f"{container_name}:/app/dist/harness", str(linux_exe)],
        cwd=root,
    )
    
    # Clean up container
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    
    if result.returncode != 0:
        print("Failed to copy executable from container.")
        return False
    
    if linux_exe.exists():
        size_mb = linux_exe.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Build successful!")
        print(f"Executable: {linux_exe}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"{'='*60}\n")
        return True
    else:
        print("Executable not found after build.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build harness executable")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before building")
    parser.add_argument("--clean-only", action="store_true", help="Only clean, don't build")
    parser.add_argument("--onedir", action="store_true", help="Build as directory instead of single file")
    parser.add_argument("--debug", action="store_true", help="Build with debug info")
    parser.add_argument("--linux", action="store_true", help="Build Linux executable via Docker")
    args = parser.parse_args()
    
    root = Path(__file__).parent.absolute()
    
    # Clean if requested
    if args.clean or args.clean_only:
        clean_build_artifacts(root)
        if args.clean_only:
            print("Clean complete.")
            return 0
    
    # Docker build for Linux
    if args.linux:
        success = build_linux_via_docker(root)
        return 0 if success else 1
    
    # Native build - check PyInstaller
    if not check_pyinstaller():
        print("PyInstaller not found. Install it with:")
        print("  pip install pyinstaller")
        return 1
    
    # Build
    success = build_executable(
        root=root,
        onefile=not args.onedir,
        debug=args.debug,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
