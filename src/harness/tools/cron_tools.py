"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

async def cron_create(self, params: Dict[str, str]) -> str:
    """Schedule a cron task (delegates to cron_tool_handlers)."""
    return await cron_create(self, params)


async def cron_delete(self, params: Dict[str, str]) -> str:
    """Cancel a cron task (delegates to cron_tool_handlers)."""
    return await cron_delete(self, params)


async def cron_list(self, params: Optional[Dict[str, str]] = None) -> str:
    """List cron tasks (delegates to cron_tool_handlers)."""
    return await cron_list(self, params)

