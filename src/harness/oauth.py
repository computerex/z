"""OAuth authentication support for harness providers.

Implements OpenAI OAuth exactly like opencode:
1. Browser-based (PKCE Authorization Code Flow)
2. Device Code Flow (Headless)
"""

import json
import time
import secrets
import hashlib
import base64
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import requests


# OpenAI OAuth Configuration (from opencode)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
CODEX_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
OAUTH_PORT = 1455
OAUTH_POLLING_SAFETY_MARGIN_MS = 3000


@dataclass
class OAuthToken:
    """OAuth token information."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: Optional[str] = None
    provider: Optional[str] = None  # "openai" or "github-copilot"
    enterprise_url: Optional[str] = None  # For GitHub Enterprise

    def is_expired(self) -> bool:
        """Check if token is expired.

        Returns False if expires_at is 0 or None (token doesn't expire).
        """
        if self.expires_at == 0 or self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account_id": self.account_id,
            "provider": self.provider,
            "enterprise_url": self.enterprise_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthToken":
        """Create from dictionary."""
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            account_id=data.get("account_id"),
            provider=data.get("provider"),
            enterprise_url=data.get("enterprise_url"),
        )


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


def generate_state() -> str:
    """Generate random state for CSRF protection."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")


def parse_jwt_claims(token: str) -> Optional[Dict[str, Any]]:
    """Parse JWT claims from token."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        # Add padding if needed
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        return json.loads(base64.urlsafe_b64decode(payload).decode())
    except Exception:
        return None


def extract_account_id(tokens: Dict[str, str]) -> Optional[str]:
    """Extract account ID from tokens."""
    # Try id_token first
    if "id_token" in tokens:
        claims = parse_jwt_claims(tokens["id_token"])
        if claims:
            return (
                claims.get("chatgpt_account_id")
                or claims.get("https://api.openai.com/auth", {}).get(
                    "chatgpt_account_id"
                )
                or claims.get("organizations", [{}])[0].get("id")
            )
    # Try access_token
    if "access_token" in tokens:
        claims = parse_jwt_claims(tokens["access_token"])
        if claims:
            return claims.get("chatgpt_account_id")
    return None


# HTML responses (from opencode)
HTML_SUCCESS = """<!doctype html>
<html>
  <head>
    <title>Harness - OpenAI Authorization Successful</title>
    <style>
      body {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #131010;
        color: #f1ecec;
      }
      .container {
        text-align: center;
        padding: 2rem;
      }
      h1 {
        color: #4ade80;
        margin-bottom: 1rem;
      }
      p {
        color: #b7b1b1;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>✓ Authorization Successful</h1>
      <p>You can close this window and return to harness.</p>
    </div>
    <script>
      setTimeout(() => window.close(), 2000)
    </script>
  </body>
</html>"""

HTML_ERROR = lambda error: (
    f"""<!doctype html>
<html>
  <head>
    <title>Harness - OpenAI Authorization Failed</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #131010;
        color: #f1ecec;
      }}
      .container {{
        text-align: center;
        padding: 2rem;
      }}
      h1 {{
        color: #fc533a;
        margin-bottom: 1rem;
      }}
      .error {{
        color: #ff917b;
        font-family: monospace;
        margin-top: 1rem;
        padding: 1rem;
        background: #3c140d;
        border-radius: 0.5rem;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>✗ Authorization Failed</h1>
      <div class="error">{error}</div>
    </div>
  </body>
</html>"""
)


class OpenAIOAuthBrowser:
    """OpenAI OAuth using browser-based PKCE flow (Method 1)."""

    _server = None
    _pending = None

    @classmethod
    def start_server(cls) -> str:
        """Start local OAuth callback server."""
        if cls._server:
            return f"http://localhost:{OAUTH_PORT}/auth/callback"

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)

                if url.path == "/auth/callback":
                    params = parse_qs(url.query)
                    code = params.get("code", [None])[0]
                    state = params.get("state", [None])[0]
                    error = params.get("error", [None])[0]
                    error_description = params.get("error_description", [None])[0]

                    if error:
                        error_msg = error_description or error
                        if cls._pending:
                            cls._pending["reject"](Exception(error_msg))
                            cls._pending = None
                        self._send_html(HTML_ERROR(error_msg), 400)
                        return

                    if not code:
                        error_msg = "Missing authorization code"
                        if cls._pending:
                            cls._pending["reject"](Exception(error_msg))
                            cls._pending = None
                        self._send_html(HTML_ERROR(error_msg), 400)
                        return

                    if not cls._pending or state != cls._pending["state"]:
                        error_msg = "Invalid state - potential CSRF attack"
                        if cls._pending:
                            cls._pending["reject"](Exception(error_msg))
                            cls._pending = None
                        self._send_html(HTML_ERROR(error_msg), 400)
                        return

                    # Exchange code for tokens
                    current = cls._pending
                    cls._pending = None

                    try:
                        tokens = cls._exchange_code(code, current["verifier"])
                        current["resolve"](tokens)
                        self._send_html(HTML_SUCCESS)
                    except Exception as e:
                        current["reject"](e)
                        self._send_html(HTML_ERROR(str(e)), 500)
                    return

                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")

            def _send_html(self, html: str, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def log_message(self, format, *args):
                pass  # Suppress logs

        # Allow socket reuse to prevent "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        cls._server = socketserver.TCPServer(("localhost", OAUTH_PORT), CallbackHandler)

        def serve():
            cls._server.serve_forever()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        return f"http://localhost:{OAUTH_PORT}/auth/callback"

    @classmethod
    def stop_server(cls):
        """Stop OAuth server (non-blocking)."""
        if cls._server:
            try:
                # Shutdown can block, so run it in a thread
                import threading

                server = cls._server
                cls._server = None
                cls._pending = None

                def do_shutdown():
                    try:
                        server.shutdown()
                        server.server_close()
                    except Exception:
                        pass

                # Run shutdown in background thread to avoid blocking
                threading.Thread(target=do_shutdown, daemon=True).start()
            except Exception:
                cls._server = None
                cls._pending = None

    @classmethod
    def _exchange_code(cls, code: str, verifier: str) -> Dict[str, str]:
        """Exchange authorization code for tokens."""
        response = requests.post(
            f"{ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": f"http://localhost:{OAUTH_PORT}/auth/callback",
                "client_id": CLIENT_ID,
                "code_verifier": verifier,
            },
            timeout=30,
        )

        if not response.ok:
            raise Exception(f"Token exchange failed: {response.status_code}")

        return response.json()

    @classmethod
    def authenticate(cls, timeout: int = 300) -> Optional[OAuthToken]:
        """Authenticate using browser-based PKCE flow."""
        print("\n  OpenAI ChatGPT Plus/Pro Authentication")
        print("  Opening browser for authorization...\n")

        try:
            # Start server
            redirect_uri = cls.start_server()

            # Generate PKCE
            verifier, challenge = generate_pkce()
            state = generate_state()

            # Build authorization URL
            auth_url = (
                f"{ISSUER}/oauth/authorize?"
                f"response_type=code&"
                f"client_id={CLIENT_ID}&"
                f"redirect_uri={redirect_uri}&"
                f"scope=openid profile email offline_access&"
                f"code_challenge={challenge}&"
                f"code_challenge_method=S256&"
                f"id_token_add_organizations=true&"
                f"codex_cli_simplified_flow=true&"
                f"state={state}&"
                f"originator=harness"
            )

            # Open browser
            webbrowser.open(auth_url)

            # Wait for callback
            result = {}
            exception = None

            def resolve(tokens):
                result["tokens"] = tokens

            def reject(error):
                nonlocal exception
                exception = error

            cls._pending = {
                "state": state,
                "verifier": verifier,
                "resolve": resolve,
                "reject": reject,
            }

            # Wait for result
            start_time = time.time()
            while time.time() - start_time < timeout:
                if "tokens" in result:
                    tokens = result["tokens"]
                    account_id = extract_account_id(tokens)

                    print(f"  ✓ Authentication successful!\n")

                    return OAuthToken(
                        access_token=tokens["access_token"],
                        refresh_token=tokens["refresh_token"],
                        expires_at=time.time() + tokens.get("expires_in", 3600),
                        account_id=account_id,
                        provider="openai",
                    )

                if exception:
                    raise exception

                time.sleep(0.1)

            raise TimeoutError("Authorization timed out")

        except Exception as e:
            print(f"  ✗ Authentication failed: {e}\n")
            return None
        finally:
            cls.stop_server()


class OpenAIOAuthDevice:
    """OpenAI OAuth using device code flow (Method 2)."""

    @classmethod
    def authenticate(cls, timeout: int = 300) -> Optional[OAuthToken]:
        """Authenticate using device code flow."""
        print("\n  OpenAI ChatGPT Plus/Pro Authentication (Headless)")
        print("  Initiating device authorization...\n")

        try:
            # Step 1: Request device code
            response = requests.post(
                f"{ISSUER}/api/accounts/deviceauth/usercode",
                headers={
                    "Content-Type": "application/json",
                },
                json={"client_id": CLIENT_ID},
                timeout=30,
            )

            if not response.ok:
                print(f"  ✗ Failed to initiate device authorization")
                return None

            data = response.json()
            device_auth_id = data["device_auth_id"]
            user_code = data["user_code"]
            interval = max(int(data.get("interval", "5")), 1)

            # Step 2: Show user code and instructions
            print(f"  Authentication required:")
            print(f"  1. Visit: {ISSUER}/codex/device")
            print(f"  2. Enter code: {user_code}\n")

            webbrowser.open(f"{ISSUER}/codex/device")

            # Step 3: Poll for token
            print("  Waiting for authorization...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                time.sleep(interval + OAUTH_POLLING_SAFETY_MARGIN_MS / 1000)

                poll_response = requests.post(
                    f"{ISSUER}/api/accounts/deviceauth/token",
                    headers={"Content-Type": "application/json"},
                    json={
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    },
                    timeout=30,
                )

                if poll_response.ok:
                    poll_data = poll_response.json()
                    auth_code = poll_data.get("authorization_code")
                    code_verifier = poll_data.get("code_verifier")

                    if auth_code and code_verifier:
                        # Exchange for tokens
                        token_response = requests.post(
                            f"{ISSUER}/oauth/token",
                            headers={
                                "Content-Type": "application/x-www-form-urlencoded"
                            },
                            data={
                                "grant_type": "authorization_code",
                                "code": auth_code,
                                "redirect_uri": f"{ISSUER}/deviceauth/callback",
                                "client_id": CLIENT_ID,
                                "code_verifier": code_verifier,
                            },
                            timeout=30,
                        )

                        if not token_response.ok:
                            print(f"  ✗ Token exchange failed")
                            return None

                        tokens = token_response.json()
                        account_id = extract_account_id(tokens)

                        print(f"  ✓ Authentication successful!\n")

                        return OAuthToken(
                            access_token=tokens["access_token"],
                            refresh_token=tokens["refresh_token"],
                            expires_at=time.time() + tokens.get("expires_in", 3600),
                            account_id=account_id,
                        )

                elif poll_response.status_code not in (403, 404):
                    # Not pending, actual error
                    print(f"  ✗ Authorization failed")
                    return None

                # Continue polling on 403/404 (authorization_pending)

            print(f"  ✗ Authorization timed out")
            return None

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            return None


class GitHubCopilotOAuth:
    """GitHub Copilot OAuth using device code flow."""

    # Default CLIENT_ID is opencode's app, but users can set their own via env var
    # To customize the OAuth screen (show your app name instead of "OpenCode by Anomaly"):
    # 1. Create a GitHub OAuth App at https://github.com/settings/developers
    # 2. Set GITHUB_COPILOT_CLIENT_ID environment variable to your app's CLIENT_ID
    DEFAULT_CLIENT_ID = "Ov23li8tweQw6odWQebz"
    GITHUB_DOMAIN = "github.com"

    @classmethod
    def get_client_id(cls) -> str:
        """Get CLIENT_ID from environment or use default."""
        import os

        return os.environ.get("GITHUB_COPILOT_CLIENT_ID", cls.DEFAULT_CLIENT_ID)

    @classmethod
    def authenticate(
        cls, enterprise_url: Optional[str] = None, timeout: int = 300
    ) -> Optional[OAuthToken]:
        """Authenticate using device code flow."""
        domain = cls.GITHUB_DOMAIN
        if enterprise_url:
            # Normalize enterprise URL
            domain = (
                enterprise_url.replace("https://", "")
                .replace("http://", "")
                .rstrip("/")
            )

        device_code_url = f"https://{domain}/login/device/code"
        access_token_url = f"https://{domain}/login/oauth/access_token"

        print("\n  GitHub Copilot Authentication")
        print(f"  Domain: {domain}\n")

        try:
            # Step 1: Request device code
            response = requests.post(
                device_code_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"client_id": cls.get_client_id(), "scope": "read:user"},
                timeout=30,
            )

            if not response.ok:
                print(f"  ✗ Failed to initiate device authorization")
                return None

            device_data = response.json()
            verification_uri = device_data.get(
                "verification_uri", f"https://{domain}/login/device"
            )
            user_code = device_data["user_code"]
            device_code = device_data["device_code"]
            interval = device_data.get("interval", 5)

            # Step 2: Show user code and instructions
            print(f"  Authentication required:")
            print(f"  1. Visit: {verification_uri}")
            print(f"  2. Enter code: {user_code}\n")

            # Try to open browser
            try:
                webbrowser.open(verification_uri)
            except Exception:
                pass

            # Step 3: Poll for token
            print("  Waiting for authorization...")
            start_time = time.time()

            while time.time() - start_time < timeout:
                time.sleep(interval + OAUTH_POLLING_SAFETY_MARGIN_MS / 1000)

                token_response = requests.post(
                    access_token_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json={
                        "client_id": cls.get_client_id(),
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    timeout=30,
                )

                if not token_response.ok:
                    return None

                token_data = token_response.json()

                if "access_token" in token_data:
                    access_token = token_data["access_token"]
                    print(f"  ✓ Authentication successful!\n")

                    return OAuthToken(
                        access_token=access_token,
                        refresh_token=access_token,  # GitHub uses same token
                        expires_at=0,  # No expiration
                        provider="github-copilot",
                        enterprise_url=enterprise_url,
                    )

                error = token_data.get("error")

                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    # Increase interval by 5 seconds as per RFC
                    interval += 5
                    if "interval" in token_data:
                        interval = token_data["interval"]
                    continue
                elif error:
                    print(f"  ✗ Authorization failed: {error}")
                    return None

            print(f"  ✗ Authorization timed out")
            return None

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            return None


class OAuthManager:
    """Manages OAuth authentication for providers."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".config" / "harness" / "oauth"
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._tokens: Dict[str, OAuthToken] = {}
        self._load_tokens()

    def _get_token_file(self, provider: str) -> Path:
        return self.config_dir / f"{provider}_token.json"

    def _load_tokens(self):
        for token_file in self.config_dir.glob("*_token.json"):
            provider = token_file.stem.replace("_token", "")
            try:
                data = json.loads(token_file.read_text())
                self._tokens[provider] = OAuthToken.from_dict(data)
            except Exception:
                pass

    def _save_token(self, provider: str, token: OAuthToken):
        token_file = self._get_token_file(provider)
        token_file.write_text(json.dumps(token.to_dict(), indent=2))

    def get_token(self, provider: str) -> Optional[OAuthToken]:
        token = self._tokens.get(provider)
        if token and token.is_expired():
            return None
        return token

    def authenticate(
        self,
        provider: str,
        method: str = "browser",
        timeout: int = 300,
        enterprise_url: Optional[str] = None,
    ) -> Optional[OAuthToken]:
        """Authenticate with a provider using OAuth.

        Args:
            provider: Provider name ("openai", "github-copilot")
            method: "browser" or "device"
            timeout: Timeout in seconds
            enterprise_url: GitHub Enterprise URL (for github-copilot-enterprise)
        """
        if provider == "openai":
            if method == "browser":
                token = OpenAIOAuthBrowser.authenticate(timeout)
            else:
                token = OpenAIOAuthDevice.authenticate(timeout)
        elif provider == "github-copilot":
            # GitHub Copilot only supports device code flow
            token = GitHubCopilotOAuth.authenticate(enterprise_url, timeout)
        else:
            print(f"  ✗ Unknown OAuth provider: {provider}")
            return None

        if token:
            self._tokens[provider] = token
            self._save_token(provider, token)

        return token

    def clear_token(self, provider: str):
        if provider in self._tokens:
            del self._tokens[provider]
        token_file = self._get_token_file(provider)
        if token_file.exists():
            token_file.unlink()


_oauth_manager: Optional[OAuthManager] = None


def get_oauth_manager() -> OAuthManager:
    global _oauth_manager
    if _oauth_manager is None:
        _oauth_manager = OAuthManager()
    return _oauth_manager
