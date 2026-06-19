"""Tool handler implementations for scheduled tasks (CronCreate / CronDelete / CronList).

These functions are designed to be mixed into ``ToolHandlers`` — the caller
passes ``self`` (the ToolHandlers instance) as the first argument so they
can access ``workspace_path`` and other dependencies.
"""

import time
from typing import Any, Dict, List, Optional

from .cron import cron_to_human, next_cron_run_ms, parse_cron_expression
from .cron_tasks import (
    add_cron_task,
    get_cron_file_path,
    list_all_cron_tasks,
    remove_cron_tasks,
)

_MAX_CRON_JOBS = 50
_MAX_AGE_DAYS = 7


# ── CronCreate ───────────────────────────────────────────────────────────────


async def cron_create(  # noqa: C901  (complexity is inherent)
    self: Any,
    params: Dict[str, str],
) -> str:
    """Schedule a prompt to run at a future time.

    Parameters
    ----------
    cron : str
        Standard 5-field cron expression in local time (e.g. ``"0 9 * * *"``).
    prompt : str
        The prompt text to enqueue when the task fires.
    recurring : str, optional
        ``"true"`` (default) = fire on every cron match.
        ``"false"`` = fire once, then auto-delete.
    durable : str, optional
        ``"true"`` = persist to ``.claude/scheduled_tasks.json``.
        ``"false"`` (default) = session-only (dies with process).
    """
    cron_str = params.get("cron", "").strip()
    prompt = params.get("prompt", "").strip()
    recurring_str = params.get("recurring", "true").strip().lower()
    durable_str = params.get("durable", "false").strip().lower()

    if not cron_str:
        return "Error: 'cron' parameter is required."
    if not prompt:
        return "Error: 'prompt' parameter is required."

    # Validate cron expression
    parsed = parse_cron_expression(cron_str)
    if parsed is None:
        return (
            f"Error: Invalid cron expression '{cron_str}'. "
            "Expected 5 fields: M H DoM Mon DoW."
        )

    # Validate it matches at least once in the next year
    if next_cron_run_ms(cron_str, time.time() * 1000) is None:
        return (
            f"Error: Cron expression '{cron_str}' does not match "
            "any calendar date in the next year."
        )

    # Check max jobs
    existing = list_all_cron_tasks()
    if len(existing) >= _MAX_CRON_JOBS:
        return f"Error: Too many scheduled jobs (max {_MAX_CRON_JOBS}). Cancel one first."

    recurring = recurring_str in ("true", "yes", "1", "t", "y")
    durable = durable_str in ("true", "yes", "1", "t", "y")

    # Create the task
    task_id = add_cron_task(
        cron=cron_str,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
    )

    human_schedule = cron_to_human(cron_str)
    where = "Persisted to .claude/scheduled_tasks.json" if durable else "Session-only (dies when harness exits)"

    if recurring:
        return (
            f"Scheduled recurring job **{task_id}** ({human_schedule}). "
            f"{where}. "
            f"Auto-expires after {_MAX_AGE_DAYS} days. "
            "Use **CronDelete** to cancel sooner."
        )
    else:
        return (
            f"Scheduled one-shot task **{task_id}** ({human_schedule}). "
            f"{where}. "
            "It will fire once then auto-delete."
        )


# ── CronDelete ───────────────────────────────────────────────────────────────


async def cron_delete(self: Any, params: Dict[str, str]) -> str:
    """Cancel a scheduled cron job by ID.

    Parameters
    ----------
    id : str
        Job ID returned by ``CronCreate``.
    """
    task_id = params.get("id", "").strip()
    if not task_id:
        return "Error: 'id' parameter is required."

    # Verify the task exists
    all_tasks = list_all_cron_tasks()
    task = next((t for t in all_tasks if t.id == task_id), None)
    if task is None:
        return f"Error: No scheduled job with id '{task_id}'."

    remove_cron_tasks([task_id])
    return f"Cancelled job **{task_id}**."


# ── CronList ─────────────────────────────────────────────────────────────────


async def cron_list(self: Any, params: Optional[Dict[str, str]] = None) -> str:
    """List all scheduled cron jobs."""
    all_tasks = list_all_cron_tasks()
    if not all_tasks:
        return "(no scheduled tasks)"

    lines: List[str] = []
    lines.append(f"**Scheduled Tasks** ({len(all_tasks)} total)")
    lines.append("")

    for t in all_tasks:
        schedule = cron_to_human(t.cron)
        durability = "session" if not t.durable else "durable"
        kind = "recurring" if t.recurring else "one-shot"
        prompt_preview = t.prompt[:120].replace("\n", " ")
        if len(t.prompt) > 120:
            prompt_preview += "…"
        lines.append(
            f"- **{t.id}** — {schedule}  "
            f"[dim]({kind}, {durability})[/dim]\n"
            f"  Prompt: {prompt_preview}"
        )

    lines.append("")
    lines.append("Use CronDelete <id> to cancel a job.")
    return "\n".join(lines)
