"""Authentication module with intentional bugs for testing code analysis."""

import hashlib
import time
import secrets


class AuthManager:
    """Manages user authentication and sessions."""
    
    def __init__(self):
        self.users = {}       # email -> {password_hash, name, role}
        self.sessions = {}    # token -> {email, created_at, expires_at}
        self.login_attempts = {}  # email -> [timestamps]
    
    def register(self, email: str, password: str, name: str, role: str = "user"):
        """Register a new user."""
        if email in self.users:
            raise ValueError("User already exists")
        
        # BUG 1: Using MD5 for password hashing (insecure)
        password_hash = hashlib.md5(password.encode()).hexdigest()
        
        self.users[email] = {
            "password_hash": password_hash,
            "name": name,
            "role": role,
        }
        return True
    
    def login(self, email: str, password: str) -> str:
        """Authenticate user and return session token."""
        # BUG 2: No rate limiting check (login_attempts tracked but never used)
        
        user = self.users.get(email)
        if not user:
            return None  # BUG 3: Should raise or return error, not None
        
        password_hash = hashlib.md5(password.encode()).hexdigest()
        if password_hash != user["password_hash"]:
            # Track failed attempt
            self.login_attempts.setdefault(email, []).append(time.time())
            return None
        
        # Create session
        token = secrets.token_hex(32)
        self.sessions[token] = {
            "email": email,
            "created_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hour
        }
        return token
    
    def validate_session(self, token: str) -> dict:
        """Validate a session token and return user info."""
        session = self.sessions.get(token)
        if not session:
            return None
        
        # BUG 4: Expired sessions are never cleaned up
        if time.time() > session["expires_at"]:
            return None  # Returns None but doesn't delete the expired session
        
        email = session["email"]
        user = self.users[email]  # BUG 5: KeyError if user was deleted after login
        return {"email": email, "name": user["name"], "role": user["role"]}
    
    def is_admin(self, token: str) -> bool:
        """Check if token belongs to an admin user."""
        info = self.validate_session(token)
        return info["role"] == "admin"  # BUG 6: NoneType error if info is None
    
    def logout(self, token: str):
        """Invalidate a session."""
        if token in self.sessions:
            del self.sessions[token]
        # BUG 7: No return value to indicate success/failure
    
    def change_password(self, email: str, old_password: str, new_password: str):
        """Change user password."""
        user = self.users.get(email)
        if not user:
            raise ValueError("User not found")
        
        old_hash = hashlib.md5(old_password.encode()).hexdigest()
        if old_hash != user["password_hash"]:
            raise ValueError("Invalid current password")
        
        # BUG 8: No password strength validation
        # BUG 9: Existing sessions not invalidated after password change
        user["password_hash"] = hashlib.md5(new_password.encode()).hexdigest()
        return True
