"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cron.cron_tool_handlers import cron_create as _cron_create_impl
from ..cron.cron_tool_handlers import cron_delete as _cron_delete_impl
from ..cron.cron_tool_handlers import cron_list as _cron_list_impl


async def cron_create(self, params: Dict[str, str]) -> str:
    """Schedule a cron task (delegates to cron_tool_handlers)."""
    return await _cron_create_impl(self, params)


async def cron_delete(self, params: Dict[str, str]) -> str:
    """Cancel a cron task (delegates to cron_tool_handlers)."""
    return await _cron_delete_impl(self, params)


async def cron_list(self, params: Optional[Dict[str, str]] = None) -> str:
    """List cron tasks (delegates to cron_tool_handlers)."""
    return await _cron_list_impl(self, params)

