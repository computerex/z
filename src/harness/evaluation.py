"""Real-world task evaluation framework for the harness."""

import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum


class TaskCategory(Enum):
    """Categories of real-world tasks."""
    
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CODE_SEARCH = "code_search"


@dataclass
class EvalTask:
    """A real-world evaluation task."""
    
    task_id: str
    name: str
    category: TaskCategory
    description: str
    prompt: str
    setup_files: Dict[str, str]  # filename -> content
    validation_fn: Optional[Callable[[Path, str], bool]] = None
    expected_files_changed: List[str] = field(default_factory=list)
    expected_output_contains: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    time_limit_seconds: int = 300
    
    def setup_workspace(self, workspace: Path):
        """Set up the workspace with initial files."""
        workspace.mkdir(parents=True, exist_ok=True)
        
        for filepath, content in self.setup_files.items():
            file_path = workspace / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)


@dataclass
class EvalResult:
    """Result of running an evaluation task."""
    
    task_id: str
    task_name: str
    category: str
    difficulty: str
    passed: bool
    score: float  # 0.0 to 1.0
    duration_seconds: float
    agent_output: str
    files_changed: List[str]
    validation_details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "passed": self.passed,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "files_changed": self.files_changed,
            "validation_details": self.validation_details,
            "error": self.error,
        }


# ============================================================================
# Real-World Evaluation Tasks
# ============================================================================

EVAL_TASKS: List[EvalTask] = []


def register_task(task: EvalTask):
    """Register an evaluation task."""
    EVAL_TASKS.append(task)
    return task


# Task 1: Fix an off-by-one error
register_task(EvalTask(
    task_id="bug_001",
    name="Fix off-by-one error in pagination",
    category=TaskCategory.BUG_FIX,
    difficulty="easy",
    description="Fix the off-by-one error in the pagination function.",
    prompt="""There's a bug in the pagination.py file. The get_page function has an off-by-one error that causes the last item to be missing from pages.

Example:
>>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> get_page(items, page=1, page_size=3)
Expected: [1, 2, 3]
Actual: [1, 2, 3]  # correct

>>> get_page(items, page=4, page_size=3)
Expected: [10]
Actual: []  # Bug! The last item is missing

Please fix this bug in pagination.py.""",
    setup_files={
        "pagination.py": '''"""Pagination utilities."""

def get_page(items: list, page: int, page_size: int) -> list:
    """Get a page of items.
    
    Args:
        items: List of all items.
        page: Page number (1-indexed).
        page_size: Number of items per page.
    
    Returns:
        List of items for the requested page.
    """
    if page < 1:
        raise ValueError("Page must be >= 1")
    
    start = (page - 1) * page_size
    end = start + page_size - 1  # Bug: should be start + page_size
    
    return items[start:end]


def get_total_pages(total_items: int, page_size: int) -> int:
    """Calculate total number of pages."""
    return (total_items + page_size - 1) // page_size
''',
        "test_pagination.py": '''"""Tests for pagination."""
import pytest
from pagination import get_page, get_total_pages

def test_get_page_first():
    items = list(range(1, 11))
    assert get_page(items, 1, 3) == [1, 2, 3]

def test_get_page_middle():
    items = list(range(1, 11))
    assert get_page(items, 2, 3) == [4, 5, 6]

def test_get_page_last():
    items = list(range(1, 11))
    assert get_page(items, 4, 3) == [10]

def test_get_total_pages():
    assert get_total_pages(10, 3) == 4
    assert get_total_pages(9, 3) == 3
''',
    },
    expected_files_changed=["pagination.py"],
    expected_output_contains=["fix", "bug", "end"],
))


# Task 2: Add input validation
register_task(EvalTask(
    task_id="feature_001",
    name="Add email validation",
    category=TaskCategory.FEATURE_ADD,
    difficulty="easy",
    description="Add email validation to the user registration function.",
    prompt="""The user.py file has a register_user function that doesn't validate email addresses. 

Please add email validation that:
1. Checks that the email contains exactly one @ symbol
2. Checks that there's at least one character before the @
3. Checks that there's a domain after the @ with at least one dot

Invalid emails should raise a ValueError with a descriptive message.

Don't use external libraries - implement the validation yourself.""",
    setup_files={
        "user.py": '''"""User management functions."""

def register_user(username: str, email: str, password: str) -> dict:
    """Register a new user.
    
    Args:
        username: The username.
        email: The email address.
        password: The password.
    
    Returns:
        User dictionary with id and details.
    """
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    
    # TODO: Add email validation
    
    return {
        "id": hash(username) % 10000,
        "username": username,
        "email": email,
    }
''',
        "test_user.py": '''"""Tests for user module."""
import pytest
from user import register_user

def test_register_valid():
    user = register_user("testuser", "test@example.com", "password123")
    assert user["username"] == "testuser"
    assert user["email"] == "test@example.com"

def test_invalid_username():
    with pytest.raises(ValueError):
        register_user("ab", "test@example.com", "password123")

def test_invalid_password():
    with pytest.raises(ValueError):
        register_user("testuser", "test@example.com", "short")

def test_invalid_email_no_at():
    with pytest.raises(ValueError):
        register_user("testuser", "testexample.com", "password123")

def test_invalid_email_no_domain():
    with pytest.raises(ValueError):
        register_user("testuser", "test@", "password123")

def test_invalid_email_no_dot():
    with pytest.raises(ValueError):
        register_user("testuser", "test@example", "password123")
''',
    },
    expected_files_changed=["user.py"],
))


# Task 3: Refactor duplicate code
register_task(EvalTask(
    task_id="refactor_001",
    name="Extract duplicate code into helper function",
    category=TaskCategory.REFACTORING,
    difficulty="medium",
    description="Refactor the data processing code to remove duplication.",
    prompt="""The data_processor.py file has duplicate code in process_users and process_orders.
Both functions have the same validation and logging logic repeated.

Please refactor to:
1. Extract the common validation logic into a helper function
2. Extract the common logging setup into a helper function  
3. Keep the same external behavior - all tests should still pass

Make the code cleaner and more maintainable.""",
    setup_files={
        "data_processor.py": '''"""Data processing functions."""
import logging

def process_users(users: list) -> list:
    """Process a list of user records."""
    # Validation - duplicated
    if not users:
        raise ValueError("Empty input list")
    if not isinstance(users, list):
        raise TypeError("Input must be a list")
    
    # Logging setup - duplicated
    logger = logging.getLogger("user_processor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logger.addHandler(handler)
    
    logger.info(f"Processing {len(users)} users")
    
    results = []
    for user in users:
        if "name" in user and "email" in user:
            results.append({
                "name": user["name"].title(),
                "email": user["email"].lower(),
            })
    
    logger.info(f"Processed {len(results)} valid users")
    return results


def process_orders(orders: list) -> list:
    """Process a list of order records."""
    # Validation - duplicated
    if not orders:
        raise ValueError("Empty input list")
    if not isinstance(orders, list):
        raise TypeError("Input must be a list")
    
    # Logging setup - duplicated
    logger = logging.getLogger("order_processor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logger.addHandler(handler)
    
    logger.info(f"Processing {len(orders)} orders")
    
    results = []
    for order in orders:
        if "product" in order and "quantity" in order:
            results.append({
                "product": order["product"],
                "quantity": int(order["quantity"]),
                "total": order.get("price", 0) * int(order["quantity"]),
            })
    
    logger.info(f"Processed {len(results)} valid orders")
    return results
''',
        "test_data_processor.py": '''"""Tests for data processor."""
import pytest
from data_processor import process_users, process_orders

def test_process_users():
    users = [
        {"name": "john doe", "email": "JOHN@EXAMPLE.COM"},
        {"name": "jane smith", "email": "Jane@Test.org"},
    ]
    result = process_users(users)
    assert len(result) == 2
    assert result[0]["name"] == "John Doe"
    assert result[0]["email"] == "john@example.com"

def test_process_users_empty():
    with pytest.raises(ValueError):
        process_users([])

def test_process_users_invalid_type():
    with pytest.raises(TypeError):
        process_users("not a list")

def test_process_orders():
    orders = [
        {"product": "Widget", "quantity": "2", "price": 10},
        {"product": "Gadget", "quantity": "3", "price": 20},
    ]
    result = process_orders(orders)
    assert len(result) == 2
    assert result[0]["total"] == 20
    assert result[1]["total"] == 60

def test_process_orders_empty():
    with pytest.raises(ValueError):
        process_orders([])
''',
    },
    expected_files_changed=["data_processor.py"],
))


# Task 4: Debug a failing test
register_task(EvalTask(
    task_id="debug_001",
    name="Debug failing JSON parser test",
    category=TaskCategory.DEBUGGING,
    difficulty="medium",
    description="Find and fix why the JSON parser tests are failing.",
    prompt="""The tests for json_parser.py are failing. Run the tests to see what's wrong:

```
python -m pytest test_json_parser.py -v
```

Debug the issue and fix the code so all tests pass. The bug is in json_parser.py, not in the tests.""",
    setup_files={
        "json_parser.py": '''"""Simple JSON parser utilities."""
import json

def parse_json_file(filepath: str) -> dict:
    """Parse a JSON file and return its contents."""
    with open(filepath, "r") as f:
        return json.load(f)


def safe_get(data: dict, path: str, default=None):
    """Safely get a nested value from a dict using dot notation.
    
    Example:
        >>> data = {"user": {"name": "John", "age": 30}}
        >>> safe_get(data, "user.name")
        "John"
        >>> safe_get(data, "user.email", "N/A")
        "N/A"
    """
    keys = path.split(".")
    result = data
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result.get(key)
        else:
            return default
    
    # Bug: returning default instead of result
    return default


def merge_json(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_json(result[key], value)
        else:
            result[key] = value
    
    return result
''',
        "test_json_parser.py": '''"""Tests for JSON parser utilities."""
import pytest
import json
import tempfile
from pathlib import Path
from json_parser import parse_json_file, safe_get, merge_json

def test_parse_json_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value"}, f)
        f.flush()
        result = parse_json_file(f.name)
    assert result == {"key": "value"}

def test_safe_get_simple():
    data = {"name": "John"}
    assert safe_get(data, "name") == "John"

def test_safe_get_nested():
    data = {"user": {"profile": {"name": "John"}}}
    assert safe_get(data, "user.profile.name") == "John"

def test_safe_get_missing():
    data = {"user": {"name": "John"}}
    assert safe_get(data, "user.email", "default") == "default"

def test_safe_get_missing_no_default():
    data = {"user": {"name": "John"}}
    assert safe_get(data, "user.email") is None

def test_merge_json_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = merge_json(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}

def test_merge_json_nested():
    base = {"user": {"name": "John", "age": 30}}
    override = {"user": {"age": 31}}
    result = merge_json(base, override)
    assert result == {"user": {"name": "John", "age": 31}}
''',
    },
    expected_files_changed=["json_parser.py"],
))


# Task 5: Add unit tests for existing code
register_task(EvalTask(
    task_id="testing_001",
    name="Write unit tests for calculator",
    category=TaskCategory.TESTING,
    difficulty="easy",
    description="Write comprehensive unit tests for the calculator module.",
    prompt="""The calculator.py module has no tests. Please create a test file test_calculator.py with comprehensive tests.

Requirements:
1. Test all four operations (add, subtract, multiply, divide)
2. Test edge cases (zero, negative numbers, large numbers)
3. Test error cases (division by zero should raise ValueError)
4. Aim for at least 10 test cases
5. Use pytest

Make sure all tests pass!""",
    setup_files={
        "calculator.py": '''"""A simple calculator module."""

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b.
    
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''',
    },
    expected_files_changed=["test_calculator.py"],
))


# Task 6: Search and document code
register_task(EvalTask(
    task_id="search_001",
    name="Find all TODO comments and create issue list",
    category=TaskCategory.CODE_SEARCH,
    difficulty="easy",
    description="Search the codebase for TODO comments and document them.",
    prompt="""Search through all Python files in this project and find all TODO comments.

Create a file called TODOS.md that lists:
1. The file path where each TODO was found
2. The line number
3. The TODO text
4. A suggested priority (high/medium/low) based on the content

Format it as a nice Markdown document.""",
    setup_files={
        "app/main.py": '''"""Main application."""
from app.utils import helper

def main():
    # TODO: Add proper argument parsing
    print("Starting app...")
    
    # TODO: Load configuration from file
    config = {}
    
    # TODO (HIGH PRIORITY): Add authentication before release!
    user = None
    
    helper()


if __name__ == "__main__":
    main()
''',
        "app/utils.py": '''"""Utility functions."""

def helper():
    """A helper function."""
    # TODO: This function needs better documentation
    pass


def process_data(data):
    """Process some data."""
    # TODO: Implement caching for better performance
    result = []
    for item in data:
        result.append(transform(item))
    return result


def transform(item):
    # TODO: Handle edge cases
    return item.upper() if isinstance(item, str) else item
''',
        "app/database.py": '''"""Database functions."""

# TODO: Replace with actual database connection
DATABASE = {}

def get_connection():
    """Get database connection."""
    # TODO (CRITICAL): Use connection pooling in production
    return DATABASE

def save_record(key, value):
    """Save a record."""
    DATABASE[key] = value
    # TODO: Add logging for all database operations
''',
    },
    expected_files_changed=["TODOS.md"],
))


# Task 7: Implement an interface
register_task(EvalTask(
    task_id="feature_002",
    name="Implement a cache interface",
    category=TaskCategory.FEATURE_ADD,
    difficulty="hard",
    description="Implement an in-memory cache with TTL support.",
    prompt="""Implement a Cache class in cache.py based on the following interface:

```python
class Cache:
    def __init__(self, default_ttl: int = 300):
        '''Initialize cache with default TTL in seconds.'''
        
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        '''Set a value with optional custom TTL.'''
        
    def get(self, key: str, default: Any = None) -> Any:
        '''Get a value, returning default if not found or expired.'''
        
    def delete(self, key: str) -> bool:
        '''Delete a key. Return True if existed, False otherwise.'''
        
    def clear(self) -> None:
        '''Clear all entries.'''
        
    def size(self) -> int:
        '''Return number of non-expired entries.'''
```

Requirements:
1. Values should expire after their TTL
2. Expired values should be cleaned up on access
3. Thread-safety is NOT required for this implementation
4. Write comprehensive tests in test_cache.py
5. All tests should pass""",
    setup_files={
        "cache.py": '''"""Cache implementation goes here."""

# TODO: Implement the Cache class
''',
    },
    expected_files_changed=["cache.py", "test_cache.py"],
))


# Task 8: Fix a complex multi-file bug
register_task(EvalTask(
    task_id="bug_002",
    name="Fix authentication flow bug",
    category=TaskCategory.BUG_FIX,
    difficulty="hard",
    description="Fix a bug in the authentication flow that spans multiple files.",
    prompt="""There's a bug in the authentication system. Users are reporting that they get logged out immediately after logging in.

The relevant files are:
- auth/session.py - Session management
- auth/login.py - Login logic
- auth/middleware.py - Request authentication

Debug the issue by examining these files, finding the bug, and fixing it.

Hint: The bug is related to how the session is being stored or validated.""",
    setup_files={
        "auth/__init__.py": '''"""Authentication package."""
''',
        "auth/session.py": '''"""Session management."""
import time
import secrets

# Simple in-memory session store
_sessions = {}

def create_session(user_id: str, duration: int = 3600) -> str:
    """Create a new session for a user.
    
    Args:
        user_id: The user's ID.
        duration: Session duration in seconds.
    
    Returns:
        Session token.
    """
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "user_id": user_id,
        "created_at": time.time(),
        "expires_at": time.time() + duration,
    }
    return token


def validate_session(token: str) -> dict | None:
    """Validate a session token.
    
    Returns:
        Session data if valid, None otherwise.
    """
    session = _sessions.get(token)
    if not session:
        return None
    
    # Bug: comparing with created_at instead of expires_at
    if time.time() > session["created_at"]:
        # Session expired
        del _sessions[token]
        return None
    
    return session


def destroy_session(token: str) -> bool:
    """Destroy a session."""
    if token in _sessions:
        del _sessions[token]
        return True
    return False
''',
        "auth/login.py": '''"""Login logic."""
from auth.session import create_session

# Mock user database
USERS = {
    "alice": {"password": "password123", "name": "Alice Smith"},
    "bob": {"password": "secret456", "name": "Bob Jones"},
}


def login(username: str, password: str) -> str | None:
    """Attempt to log in a user.
    
    Args:
        username: The username.
        password: The password.
    
    Returns:
        Session token if successful, None otherwise.
    """
    user = USERS.get(username)
    if not user:
        return None
    
    if user["password"] != password:
        return None
    
    # Create session and return token
    token = create_session(username)
    return token
''',
        "auth/middleware.py": '''"""Authentication middleware."""
from auth.session import validate_session


def authenticate_request(token: str) -> dict | None:
    """Authenticate a request using a session token.
    
    Args:
        token: The session token from the request.
    
    Returns:
        User info if authenticated, None otherwise.
    """
    session = validate_session(token)
    if not session:
        return None
    
    return {
        "user_id": session["user_id"],
        "authenticated": True,
    }
''',
        "test_auth.py": '''"""Tests for authentication."""
import time
from auth.login import login
from auth.middleware import authenticate_request
from auth.session import create_session, validate_session, destroy_session


def test_login_success():
    token = login("alice", "password123")
    assert token is not None


def test_login_wrong_password():
    token = login("alice", "wrongpassword")
    assert token is None


def test_login_unknown_user():
    token = login("unknown", "password")
    assert token is None


def test_session_valid():
    token = create_session("testuser")
    session = validate_session(token)
    assert session is not None
    assert session["user_id"] == "testuser"


def test_full_auth_flow():
    """Test the complete authentication flow."""
    # Login
    token = login("alice", "password123")
    assert token is not None
    
    # Authenticate request
    auth_result = authenticate_request(token)
    assert auth_result is not None
    assert auth_result["user_id"] == "alice"
    assert auth_result["authenticated"] is True


def test_destroyed_session():
    token = create_session("testuser")
    destroy_session(token)
    session = validate_session(token)
    assert session is None
''',
    },
    expected_files_changed=["auth/session.py"],
))


class EvalRunner:
    """Runner for evaluation tasks."""
    
    def __init__(self, agent, workspace_base: Optional[Path] = None):
        """Initialize the evaluation runner."""
        self.agent = agent
        self.workspace_base = workspace_base or Path(tempfile.gettempdir()) / "eval_tasks"
        self.results: List[EvalResult] = []
    
    def _check_files_changed(self, workspace: Path, initial_hashes: Dict[str, str]) -> List[str]:
        """Check which files were changed."""
        changed = []
        
        for filepath in workspace.rglob("*"):
            if filepath.is_file():
                rel_path = str(filepath.relative_to(workspace))
                current_hash = hash(filepath.read_text())
                
                if rel_path not in initial_hashes:
                    changed.append(rel_path)  # New file
                elif current_hash != initial_hashes[rel_path]:
                    changed.append(rel_path)  # Modified file
        
        return changed
    
    def _run_tests(self, workspace: Path) -> bool:
        """Run pytest in the workspace."""
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "pytest", "-v", "--tb=short"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        return result.returncode == 0
    
    async def run_task(self, task: EvalTask) -> EvalResult:
        """Run a single evaluation task."""
        start_time = datetime.now()
        workspace = self.workspace_base / task.task_id
        
        # Clean up and set up workspace
        if workspace.exists():
            shutil.rmtree(workspace)
        task.setup_workspace(workspace)
        
        # Hash initial files
        initial_hashes = {}
        for filepath in workspace.rglob("*"):
            if filepath.is_file():
                rel_path = str(filepath.relative_to(workspace))
                initial_hashes[rel_path] = hash(filepath.read_text())
        
        try:
            # Update agent workspace
            self.agent.config.workspace_path = workspace
            
            # Run agent
            agent_output = await asyncio.wait_for(
                self.agent.run(task.prompt),
                timeout=task.time_limit_seconds,
            )
            
            # Check results
            files_changed = self._check_files_changed(workspace, initial_hashes)
            
            # Run tests if available
            test_passed = True
            if list(workspace.glob("test_*.py")):
                test_passed = self._run_tests(workspace)
            
            # Calculate score
            score = 0.0
            validation_details = {}
            
            # Check expected files changed
            expected_changed = set(task.expected_files_changed)
            actual_changed = set(files_changed)
            
            if expected_changed:
                file_score = len(expected_changed & actual_changed) / len(expected_changed)
                score += file_score * 0.3
                validation_details["expected_files_changed"] = list(expected_changed)
                validation_details["actual_files_changed"] = list(actual_changed)
            
            # Check tests pass
            if test_passed:
                score += 0.5
            validation_details["tests_passed"] = test_passed
            
            # Check expected output
            if task.expected_output_contains:
                matches = sum(1 for s in task.expected_output_contains if s.lower() in agent_output.lower())
                output_score = matches / len(task.expected_output_contains)
                score += output_score * 0.2
            
            passed = test_passed and score >= 0.5
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return EvalResult(
                task_id=task.task_id,
                task_name=task.name,
                category=task.category.value,
                difficulty=task.difficulty,
                passed=passed,
                score=score,
                duration_seconds=duration,
                agent_output=agent_output,
                files_changed=files_changed,
                validation_details=validation_details,
            )
        
        except asyncio.TimeoutError:
            return EvalResult(
                task_id=task.task_id,
                task_name=task.name,
                category=task.category.value,
                difficulty=task.difficulty,
                passed=False,
                score=0.0,
                duration_seconds=task.time_limit_seconds,
                agent_output="",
                files_changed=[],
                error="Task timed out",
            )
        
        except Exception as e:
            return EvalResult(
                task_id=task.task_id,
                task_name=task.name,
                category=task.category.value,
                difficulty=task.difficulty,
                passed=False,
                score=0.0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                agent_output="",
                files_changed=[],
                error=str(e),
            )
    
    async def run_all(
        self,
        tasks: Optional[List[EvalTask]] = None,
        categories: Optional[List[TaskCategory]] = None,
        difficulties: Optional[List[str]] = None,
    ) -> List[EvalResult]:
        """Run all matching evaluation tasks."""
        tasks = tasks or EVAL_TASKS
        
        # Filter by category
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        
        # Filter by difficulty
        if difficulties:
            tasks = [t for t in tasks if t.difficulty in difficulties]
        
        for i, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Task {i+1}/{len(tasks)}: {task.name}")
            print(f"Category: {task.category.value} | Difficulty: {task.difficulty}")
            print(f"{'='*60}\n")
            
            result = await self.run_task(task)
            self.results.append(result)
            
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"\nResult: {status} (score: {result.score:.2f})")
            if result.error:
                print(f"Error: {result.error}")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        if not self.results:
            return {"total": 0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        by_category = {}
        for r in self.results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0}
            by_category[cat]["total"] += 1
            if r.passed:
                by_category[cat]["passed"] += 1
        
        by_difficulty = {}
        for r in self.results:
            diff = r.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "passed": 0}
            by_difficulty[diff]["total"] += 1
            if r.passed:
                by_difficulty[diff]["passed"] += 1
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total,
            "average_score": sum(r.score for r in self.results) / total,
            "average_duration": sum(r.duration_seconds for r in self.results) / total,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }
    
    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {output_path}")


async def run_evaluation(
    config,
    categories: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    """Run the evaluation suite.
    
    Args:
        config: Agent configuration.
        categories: Task categories to run.
        difficulties: Difficulty levels to run.
        max_tasks: Maximum number of tasks.
        output_dir: Output directory for results.
    """
    from .agent import Agent
    
    output_dir = output_dir or Path("eval_results")
    
    agent = Agent(config, max_iterations=20)
    runner = EvalRunner(agent)
    
    # Parse category filter
    cat_filter = None
    if categories:
        cat_filter = [TaskCategory(c) for c in categories]
    
    tasks = EVAL_TASKS[:max_tasks] if max_tasks else EVAL_TASKS
    
    print(f"Running {len(tasks)} evaluation tasks...")
    
    await runner.run_all(
        tasks=tasks,
        categories=cat_filter,
        difficulties=difficulties,
    )
    
    summary = runner.get_summary()
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tasks: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Average duration: {summary['average_duration']:.1f}s")
    
    print("\nBy Category:")
    for cat, stats in summary.get("by_category", {}).items():
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    print("\nBy Difficulty:")
    for diff, stats in summary.get("by_difficulty", {}).items():
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    print("="*60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(output_dir / f"eval_{timestamp}.json")
    
    return summary
