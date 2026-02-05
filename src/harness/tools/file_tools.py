"""File operation tools for the harness."""

import os
import fnmatch
import aiofiles
from pathlib import Path
from typing import List, Optional, Dict, Any


async def read_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read content from a file, optionally with line range.
    
    Args:
        file_path: Path to the file to read.
        start_line: Optional 1-based start line number.
        end_line: Optional 1-based end line number (inclusive).
    
    Returns:
        File content as a string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
        content = await f.read()
    
    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else len(lines)
        content = "".join(lines[start_idx:end_idx])
    
    return content


async def write_file(file_path: str, content: str, create_dirs: bool = True) -> str:
    """Write content to a file.
    
    Args:
        file_path: Path to the file to write.
        content: Content to write.
        create_dirs: Whether to create parent directories.
    
    Returns:
        Success message.
    """
    path = Path(file_path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)
    
    return f"Successfully wrote {len(content)} bytes to {file_path}"


async def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
) -> str:
    """Edit a file by replacing a string.
    
    Args:
        file_path: Path to the file to edit.
        old_string: The exact string to replace.
        new_string: The replacement string.
    
    Returns:
        Success message with details.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    
    # Count occurrences
    count = content.count(old_string)
    
    if count == 0:
        raise ValueError(f"String not found in file: '{old_string[:100]}...'")
    
    if count > 1:
        raise ValueError(
            f"String found {count} times. Please provide more context for unique match."
        )
    
    # Perform replacement
    new_content = content.replace(old_string, new_string, 1)
    
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(new_content)
    
    return f"Successfully edited {file_path}"


async def list_directory(
    directory_path: str,
    recursive: bool = False,
    pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List contents of a directory.
    
    Args:
        directory_path: Path to the directory.
        recursive: Whether to list recursively.
        pattern: Optional glob pattern to filter results.
    
    Returns:
        List of file/directory info dictionaries.
    """
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")
    
    results = []
    
    if recursive:
        iterator = path.rglob("*")
    else:
        iterator = path.iterdir()
    
    for item in iterator:
        if pattern and not fnmatch.fnmatch(item.name, pattern):
            continue
        
        try:
            stat = item.stat()
            results.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": stat.st_size if item.is_file() else None,
            })
        except (PermissionError, OSError):
            continue
    
    return results


async def file_search(
    directory: str,
    pattern: str,
    max_results: int = 100,
) -> List[str]:
    """Search for files matching a glob pattern.
    
    Args:
        directory: Root directory to search from.
        pattern: Glob pattern to match (e.g., "**/*.py").
        max_results: Maximum number of results to return.
    
    Returns:
        List of matching file paths.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    results = []
    for match in path.glob(pattern):
        if match.is_file():
            results.append(str(match))
            if len(results) >= max_results:
                break
    
    return results
