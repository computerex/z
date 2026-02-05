"""Shell command execution tools."""

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShellResult:
    """Result of a shell command execution."""
    
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
        }


async def run_shell_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: float = 60.0,
    shell: bool = True,
) -> ShellResult:
    """Execute a shell command.
    
    Args:
        command: The command to execute.
        cwd: Working directory for the command.
        timeout: Timeout in seconds.
        shell: Whether to run in shell mode.
    
    Returns:
        ShellResult with stdout, stderr, and return code.
    """
    try:
        # Platform-specific shell handling
        if sys.platform == "win32":
            # Use PowerShell on Windows
            if shell:
                cmd = ["powershell", "-NoProfile", "-Command", command]
            else:
                cmd = command.split()
        else:
            if shell:
                cmd = command
            else:
                cmd = command.split()
        
        process = await asyncio.create_subprocess_shell(
            command if shell else " ".join(cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return ShellResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                return_code=process.returncode or 0,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ShellResult(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=-1,
                timed_out=True,
            )
    
    except Exception as e:
        return ShellResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
        )
