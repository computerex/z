"""Todo list manager for tracking agent goals and objectives.

Provides a persistent, structured todo list that captures user intent
independently of context. The agent uses this to stay grounded during
long-running autonomous sessions, and context compaction/eviction can
reference active todos to decide what to keep.
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TodoStatus(str, Enum):
    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class TodoItem:
    """A single todo item."""
    id: int
    title: str
    status: TodoStatus = TodoStatus.NOT_STARTED
    description: str = ""  # Detailed description / acceptance criteria
    parent_id: Optional[int] = None  # For sub-tasks
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    context_refs: List[str] = field(default_factory=list)  # File paths, search terms relevant to this todo
    notes: str = ""  # Agent's working notes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "description": self.description,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "context_refs": self.context_refs,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TodoItem":
        return cls(
            id=data["id"],
            title=data["title"],
            status=TodoStatus(data.get("status", "not-started")),
            description=data.get("description", ""),
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at"),
            context_refs=data.get("context_refs", []),
            notes=data.get("notes", ""),
        )

    def format_short(self) -> str:
        """One-line summary."""
        status_icons = {
            TodoStatus.NOT_STARTED: "○",
            TodoStatus.IN_PROGRESS: "◐",
            TodoStatus.COMPLETED: "●",
            TodoStatus.BLOCKED: "✗",
        }
        icon = status_icons.get(self.status, "?")
        prefix = f"  └─" if self.parent_id else ""
        return f"{prefix}[{self.id}] {icon} {self.title} ({self.status.value})"


class TodoManager:
    """Manages a structured todo list for the agent.
    
    The todo list is the source of truth for what the agent is trying to
    accomplish. It survives context compaction and can be used to:
    - Ground context eviction decisions (keep context relevant to active todos)
    - Orient the agent after context truncation
    - Track long-running autonomous progress
    - Provide user visibility into agent planning
    """

    def __init__(self):
        self._items: Dict[int, TodoItem] = {}
        self._next_id = 1
        self._original_request: str = ""  # The user's original request

    def set_original_request(self, request: str):
        """Store the original user request for grounding."""
        self._original_request = request

    @property
    def original_request(self) -> str:
        return self._original_request

    def add(
        self,
        title: str,
        description: str = "",
        parent_id: Optional[int] = None,
        context_refs: Optional[List[str]] = None,
    ) -> TodoItem:
        """Add a new todo item."""
        item = TodoItem(
            id=self._next_id,
            title=title,
            description=description,
            parent_id=parent_id,
            context_refs=context_refs or [],
        )
        self._items[item.id] = item
        self._next_id += 1
        return item

    def update(
        self,
        item_id: int,
        title: Optional[str] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
        context_refs: Optional[List[str]] = None,
    ) -> Optional[TodoItem]:
        """Update a todo item. Returns updated item or None if not found."""
        item = self._items.get(item_id)
        if not item:
            return None

        if title is not None:
            item.title = title
        if status is not None:
            item.status = TodoStatus(status)
            if item.status == TodoStatus.COMPLETED:
                item.completed_at = time.time()
        if description is not None:
            item.description = description
        if notes is not None:
            item.notes = notes
        if context_refs is not None:
            item.context_refs = context_refs

        item.updated_at = time.time()
        return item

    def remove(self, item_id: int) -> bool:
        """Remove a todo item and its children."""
        if item_id not in self._items:
            return False
        # Remove children first
        children = [id for id, item in self._items.items() if item.parent_id == item_id]
        for child_id in children:
            del self._items[child_id]
        del self._items[item_id]
        return True

    def get(self, item_id: int) -> Optional[TodoItem]:
        return self._items.get(item_id)

    def list_all(self) -> List[TodoItem]:
        """List all todos, ordered by id."""
        return sorted(self._items.values(), key=lambda x: x.id)

    def list_active(self) -> List[TodoItem]:
        """List todos that are not completed."""
        return [
            item for item in self.list_all()
            if item.status != TodoStatus.COMPLETED
        ]

    def list_in_progress(self) -> List[TodoItem]:
        """List todos currently being worked on."""
        return [
            item for item in self.list_all()
            if item.status == TodoStatus.IN_PROGRESS
        ]

    def get_active_context_refs(self) -> List[str]:
        """Get all context references from active (non-completed) todos.
        
        Used by the smart context manager to determine which context
        items are relevant to current work.
        """
        refs = []
        for item in self.list_active():
            refs.extend(item.context_refs)
        return refs

    def get_progress_summary(self) -> str:
        """Get a concise progress summary for context injection."""
        items = self.list_all()
        if not items:
            return "No todos defined."

        total = len(items)
        completed = sum(1 for i in items if i.status == TodoStatus.COMPLETED)
        in_progress = sum(1 for i in items if i.status == TodoStatus.IN_PROGRESS)
        blocked = sum(1 for i in items if i.status == TodoStatus.BLOCKED)
        not_started = total - completed - in_progress - blocked

        summary = f"Progress: {completed}/{total} complete"
        if in_progress:
            summary += f", {in_progress} in progress"
        if blocked:
            summary += f", {blocked} blocked"
        if not_started:
            summary += f", {not_started} not started"

        return summary

    def format_list(self, include_completed: bool = True) -> str:
        """Format the full todo list for display/injection into context."""
        items = self.list_all()
        if not items:
            return "Todo list is empty."

        lines = ["TODO LIST:"]
        if self._original_request:
            lines.append(f"Original request: {self._original_request[:500]}")
        lines.append(self.get_progress_summary())
        lines.append("")

        # Group by parent
        roots = [i for i in items if not i.parent_id]
        children_map: Dict[int, List[TodoItem]] = {}
        for item in items:
            if item.parent_id:
                children_map.setdefault(item.parent_id, []).append(item)

        for root in roots:
            if not include_completed and root.status == TodoStatus.COMPLETED:
                continue
            lines.append(root.format_short())
            if root.notes:
                lines.append(f"      Notes: {root.notes[:200]}")
            for child in children_map.get(root.id, []):
                if not include_completed and child.status == TodoStatus.COMPLETED:
                    continue
                lines.append(child.format_short())

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for session persistence."""
        return {
            "items": [item.to_dict() for item in self._items.values()],
            "next_id": self._next_id,
            "original_request": self._original_request,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TodoManager":
        """Deserialize from session data."""
        mgr = cls()
        mgr._original_request = data.get("original_request", "")
        mgr._next_id = data.get("next_id", 1)
        for item_data in data.get("items", []):
            item = TodoItem.from_dict(item_data)
            mgr._items[item.id] = item
        return mgr

    def clear(self):
        """Clear all todos."""
        self._items.clear()
        self._next_id = 1
        self._original_request = ""
