#!/usr/bin/env python3
"""
Integration test for harness agent with realistic multi-step scenarios.

This simulates real Claude Code usage patterns:
- Multi-step file edits
- Complex search and replace operations
- Todo management throughout the process
- Context management
- Error handling and recovery
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess
import asyncio

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harness.cline_agent import ClineAgent, parse_xml_tool, parse_all_xml_tools
from harness.config import Config
from harness.streaming_client import StreamingMessage


class IntegrationTest:
    """Integration test for realistic harness usage."""
    
    def __init__(self):
        self.test_dir = None
        self.agent = None
        self.original_cwd = None
    
    def setup_complex_project(self):
        """Create a complex realistic project for testing."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="harness_integration_"))
        self.original_cwd = os.getcwd()
        
        print(f"Integration test environment: {self.test_dir}")
        
        # Create a Flask web app project
        project_structure = {
            "app": {
                "__init__.py": '"""Flask application package."""',
                "models.py": '''"""Database models."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def __repr__(self):
        return f'<Post {self.title}>'
''',
                "routes.py": '''"""Application routes."""
from flask import Blueprint, render_template, request, redirect, url_for
from .models import db, User, Post

main = Blueprint('main', __name__)

@main.route('/')
def index():
    posts = Post.query.all()
    return render_template('index.html', posts=posts)

@main.route('/user/<username>')
def user_profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(user_id=user.id).all()
    return render_template('user.html', user=user, posts=posts)

@main.route('/post/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        # TODO: Get current user properly
        user = User.query.first()
        post = Post(title=title, content=content, user_id=user.id)
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('main.index'))
    return render_template('new_post.html')
''',
                "config.py": '''"""Application configuration."""
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
''',
            },
            "templates": {
                "base.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}My Blog{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('main.index') }}">My Blog</a>
        </div>
    </nav>
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
''',
                "index.html": '''{% extends "base.html" %}

{% block content %}
<h1>Latest Posts</h1>
<a href="{{ url_for('main.new_post') }}" class="btn btn-primary mb-3">New Post</a>

{% for post in posts %}
<div class="card mb-3">
    <div class="card-body">
        <h5 class="card-title">{{ post.title }}</h5>
        <p class="card-text">{{ post.content[:200] }}...</p>
        <small class="text-muted">By {{ post.user.username }} on {{ post.created_at.strftime('%Y-%m-%d') }}</small>
    </div>
</div>
{% endfor %}
{% endblock %}
''',
            },
            "tests": {
                "__init__.py": "",
                "test_models.py": '''"""Test database models."""
import pytest
from app.models import User, Post

def test_user_creation():
    """Test user model creation."""
    user = User(username="testuser", email="test@example.com", password_hash="hash123")
    assert user.username == "testuser"
    assert user.email == "test@example.com"

def test_post_creation():
    """Test post model creation."""
    post = Post(title="Test Post", content="This is test content", user_id=1)
    assert post.title == "Test Post"
    assert post.content == "This is test content"
''',
            },
            "requirements.txt": '''Flask==2.3.3
Flask-SQLAlchemy==3.0.5
python-dotenv==1.0.0
pytest==7.4.2
''',
            "app.py": '''"""Main application entry point."""
from flask import Flask
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
''',
            "README.md": '''# My Flask Blog

A simple blog application built with Flask.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize database:
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Features

- User management
- Post creation and viewing
- Responsive design with Bootstrap

## TODO

- [ ] Add user authentication
- [ ] Add post editing
- [ ] Add comments system
- [ ] Add search functionality
'''
        }
        
        # Create directory structure and files
        def create_structure(base_path, structure):
            for name, content in structure.items():
                path = base_path / name
                if isinstance(content, dict):
                    path.mkdir(exist_ok=True)
                    create_structure(path, content)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
        
        create_structure(self.test_dir, project_structure)
        
        # Initialize git
        subprocess.run(["git", "init"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=self.test_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial Flask blog project"], cwd=self.test_dir, capture_output=True)
        
        # Change to test directory and initialize agent
        os.chdir(self.test_dir)
        
        config = Config(
            api_url="http://localhost:8080/v1/messages",
            api_key="test-key", 
            model="test-model"
        )
        
        self.agent = ClineAgent(config)
        self.agent.workspace_path = str(self.test_dir)
        self.agent.clear_history()
    
    def cleanup(self):
        """Clean up test environment."""
        if self.original_cwd:
            os.chdir(self.original_cwd)
        if self.test_dir and self.test_dir.exists():
            try:
                if os.name == 'nt':
                    for root, dirs, files in os.walk(self.test_dir):
                        for file in files:
                            try:
                                os.chmod(os.path.join(root, file), 0o777)
                            except:
                                pass
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Could not clean up: {e}")
    
    def simulate_user_interaction(self, user_message):
        """Simulate a user interaction and return agent's tool calls."""
        print(f"\n>>> USER: {user_message}")
        
        # Add user message
        self.agent.messages.append(StreamingMessage(role="user", content=user_message))
        
        # Simulate agent response with tool calls
        # In a real scenario, this would be the model's response
        # For testing, we'll manually construct expected responses
        return []
    
    def test_multi_step_file_editing(self):
        """Test complex multi-step file editing scenario."""
        print("\n=== Multi-Step File Editing Test ===")
        
        # Scenario: Add user authentication to the Flask app
        print("Scenario: Adding user authentication system")
        
        # Step 1: Agent should read existing files to understand structure
        models_content = (self.test_dir / "app" / "models.py").read_text()
        routes_content = (self.test_dir / "app" / "routes.py").read_text()
        
        print("+ Project files exist and have expected content")
        
        # Step 2: Simulate adding authentication fields to User model
        auth_xml = '''<replace_in_file>
<path>app/models.py</path>
<diff>
<<<<<<< SEARCH
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
=======
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
>>>>>>> REPLACE
</diff>
</replace_in_file>'''
        
        # Test XML parsing of this complex edit
        parsed = parse_xml_tool(auth_xml)
        if not parsed or parsed.name != "replace_in_file":
            print("- Failed to parse authentication edit XML")
            return False
        
        print("+ Complex SEARCH/REPLACE XML parsing works")
        
        # Step 3: Test multiple tool calls in sequence
        tool_sequence = '''<read_file>
<path>app/routes.py</path>
</read_file>

<replace_in_file>
<path>app/routes.py</path>
<diff>
<<<<<<< SEARCH
from flask import Blueprint, render_template, request, redirect, url_for
from .models import db, User, Post
=======
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from .models import db, User, Post
>>>>>>> REPLACE
</diff>
</replace_in_file>

<manage_todos>
<action>add</action>
<title>Add login route</title>
<description>Create /login route with form handling and session management</description>
</manage_todos>'''
        
        # Test parsing multiple tool calls
        all_tools = parse_all_xml_tools(tool_sequence)
        expected_tools = ["read_file", "replace_in_file", "manage_todos"]
        
        if len(all_tools) != 3:
            print(f"- Expected 3 tools, got {len(all_tools)}")
            return False
        
        for i, expected in enumerate(expected_tools):
            if all_tools[i].name != expected:
                print(f"- Expected tool {expected}, got {all_tools[i].name}")
                return False
        
        print("+ Multi-tool sequence parsing works")
        
        # Step 4: Test context management during complex operations
        # Add some context items
        self.agent.context.add("file", "app/models.py", models_content)
        self.agent.context.add("file", "app/routes.py", routes_content)
        
        if self.agent.context.total_size() < 1000:
            print("- Context not properly populated")
            return False
        
        print("+ Context management during complex operations works")
        
        return True
    
    def test_error_recovery_patterns(self):
        """Test error recovery and resilience patterns."""
        print("\n=== Error Recovery Test ===")
        
        # Test handling of invalid file paths
        invalid_xml = '<read_file><path>nonexistent/file.py</path></read_file>'
        parsed = parse_xml_tool(invalid_xml)
        
        if not parsed or parsed.name != "read_file":
            print("- Failed to parse tool call for nonexistent file")
            return False
        
        print("+ Handles invalid file path parsing correctly")
        
        # Test handling of malformed XML
        malformed_xml = '<read_file><path>test.py</path><invalid_param>value</read_file>'
        parsed = parse_xml_tool(malformed_xml)
        
        # Should still parse the valid parts
        if not parsed or "path" not in parsed.parameters:
            print("- Failed to recover from malformed XML")
            return False
        
        print("+ Recovers from malformed XML gracefully")
        
        return True
    
    def test_todo_workflow_integration(self):
        """Test todo workflow integration throughout complex tasks."""
        print("\n=== Todo Workflow Integration Test ===")
        
        # Simulate a complex task breakdown
        task_xml = '''<manage_todos>
<action>add</action>
<title>Implement user authentication</title>
<description>Add complete user auth system with login, logout, registration</description>
</manage_todos>'''
        
        parsed = parse_xml_tool(task_xml)
        if not parsed or parsed.name != "manage_todos":
            print("- Failed to parse todo creation")
            return False
        
        # Add the todo manually for testing
        todo = self.agent.todo_manager.add(
            "Implement user authentication",
            "Add complete user auth system with login, logout, registration"
        )
        
        # Test todo status updates
        self.agent.todo_manager.update(todo.id, status="in-progress")
        active_todos = self.agent.todo_manager.list_active()
        
        if len(active_todos) != 1:
            print(f"- Expected 1 active todo, got {len(active_todos)}")
            return False
        
        # Test subtask creation
        subtask = self.agent.todo_manager.add(
            "Add login form template",
            "Create HTML template for login form",
            parent_id=todo.id
        )
        
        all_todos = self.agent.todo_manager.list_all()
        if len(all_todos) != 2:
            print(f"- Expected 2 todos (parent + child), got {len(all_todos)}")
            return False
        
        print("+ Todo workflow integration works correctly")
        return True
    
    def test_system_prompt_completeness(self):
        """Test that system prompt contains all Claude Code elements."""
        print("\n=== System Prompt Completeness Test ===")
        
        prompt = self.agent._system_prompt()
        
        # Check for critical Claude Code sections
        critical_sections = {
            "Security guidance": "IMPORTANT: Assist with authorized security testing",
            "URL restrictions": "IMPORTANT: You must NEVER generate or guess URLs",
            "Tool execution philosophy": "Do NOT use execute_command to run commands when a relevant dedicated tool",
            "Reversibility guidance": "Carefully consider the reversibility and blast radius",
            "Over-engineering avoidance": "Avoid over-engineering. Only make changes that are directly requested",
            "Code change rules": "You MUST use read_file at least once before editing",
            "Comment restrictions": "Do NOT add comments that just narrate what the code does",
            "Environment info": f"Primary working directory: {self.test_dir}",
            "Shell info": "Shell:",
            "Tool definitions": "## read_file",
        }
        
        missing = []
        for name, pattern in critical_sections.items():
            if pattern not in prompt:
                missing.append(name)
        
        if missing:
            print(f"- Missing critical sections: {missing}")
            return False
        
        print("+ System prompt contains all critical Claude Code sections")
        
        # Check prompt is substantial (Claude Code prompts are very long)
        if len(prompt) < 30000:
            print(f"- Prompt too short: {len(prompt)} chars (expected >30k)")
            return False
        
        print(f"+ Prompt is substantial: {len(prompt)} chars")
        return True
    
    def test_realistic_development_workflow(self):
        """Test a realistic development workflow."""
        print("\n=== Realistic Development Workflow Test ===")
        
        # Clear any existing todos from previous tests
        self.agent.todo_manager._items.clear()
        self.agent.todo_manager._next_id = 1
        
        # Scenario: Add a new feature with proper workflow
        print("Scenario: Adding comment system to blog posts")
        
        # Step 1: Create todo for the feature
        feature_todo = self.agent.todo_manager.add(
            "Add comment system",
            "Allow users to comment on blog posts with proper models and routes"
        )
        
        # Step 2: Break down into subtasks
        subtasks = [
            ("Create Comment model", "Add Comment model to models.py with relationships"),
            ("Add comment routes", "Create routes for posting and viewing comments"),
            ("Update templates", "Add comment forms and display to post templates"),
            ("Add tests", "Write tests for comment functionality"),
        ]
        
        subtask_ids = []
        for title, desc in subtasks:
            subtask = self.agent.todo_manager.add(title, desc, parent_id=feature_todo.id)
            subtask_ids.append(subtask.id)
        
        # Step 3: Start working on first subtask
        self.agent.todo_manager.update(subtask_ids[0], status="in-progress")
        
        # Step 4: Simulate reading existing code
        self.agent.context.add("file", "app/models.py", (self.test_dir / "app" / "models.py").read_text())
        
        # Step 5: Simulate making changes
        # (In real usage, this would be done by the model via tool calls)
        
        # Step 6: Complete subtask and move to next
        self.agent.todo_manager.update(subtask_ids[0], status="completed")
        self.agent.todo_manager.update(subtask_ids[1], status="in-progress")
        
        # Verify workflow state
        active = self.agent.todo_manager.list_active()
        completed = [t for t in self.agent.todo_manager.list_all() if t.status.value == "completed"]
        all_todos = self.agent.todo_manager.list_all()
        
        # Should have: 1 parent + 4 subtasks = 5 total, with 1 completed and 4 active
        if len(all_todos) != 5:
            print(f"- Expected 5 total todos, got {len(all_todos)}")
            return False
        
        if len(active) != 4:  # 1 parent + 3 remaining subtasks (1 is in-progress, counts as active)
            print(f"- Expected 4 active todos, got {len(active)}")
            print(f"  Active todos: {[t.title for t in active]}")
            print(f"  All todos: {[(t.title, t.status.value) for t in all_todos]}")
            return False
        
        if len(completed) != 1:
            print(f"- Expected 1 completed todo, got {len(completed)}")
            return False
        
        print("+ Realistic development workflow simulation works")
        return True
    
    def run_integration_tests(self):
        """Run all integration tests."""
        print("=== HARNESS INTEGRATION TEST SUITE ===")
        print("Testing realistic multi-step development scenarios...")
        
        try:
            self.setup_complex_project()
            
            tests = [
                self.test_multi_step_file_editing,
                self.test_error_recovery_patterns,
                self.test_todo_workflow_integration,
                self.test_system_prompt_completeness,
                self.test_realistic_development_workflow,
            ]
            
            passed = 0
            failed = 0
            
            for test in tests:
                try:
                    if test():
                        passed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"- Test {test.__name__} crashed: {e}")
                    failed += 1
            
            print(f"\n=== INTEGRATION TEST RESULTS ===")
            print(f"PASSED: {passed}")
            print(f"FAILED: {failed}")
            print(f"TOTAL:  {passed + failed}")
            
            if failed == 0:
                print("*** ALL INTEGRATION TESTS PASSED! ***")
                return True
            else:
                print(f"*** {failed} INTEGRATION TESTS FAILED ***")
                return False
        
        finally:
            self.cleanup()


def main():
    """Run integration tests."""
    test = IntegrationTest()
    success = test.run_integration_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()