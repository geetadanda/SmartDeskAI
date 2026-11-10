"""Local SQLite-backed authentication for the SmartHR Streamlit app."""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Iterator

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "smarthr_auth.sqlite3"
PASSWORD_HASHER = PasswordHasher()
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))
MIN_PASSWORD_LENGTH = 12
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 15 * 60


class AuthConfigurationError(RuntimeError):
    """Raised when authentication has not been configured safely."""


def _get_setting(name: str) -> str:
    """Read deployment settings from environment variables or Streamlit secrets."""
    value = os.getenv(name, "")
    if value:
        return value
    try:
        secret = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        return ""
    return str(secret) if secret is not None else ""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_auth() -> None:
    """Create the local account and login-attempt tables if they do not exist."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                session_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS login_attempts (
                username TEXT NOT NULL,
                attempted_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS login_attempts_by_user "
            "ON login_attempts(username, attempted_at)"
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(users)")
        }
        if "session_version" not in columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN session_version "
                "INTEGER NOT NULL DEFAULT 0"
            )
    try:
        DATABASE_PATH.chmod(0o600)
    except OSError as exc:
        raise AuthConfigurationError(
            f"Could not restrict permissions on the account database: {exc}"
        ) from exc


def ensure_initial_admin() -> bool:
    """Create the first admin only when both bootstrap environment values exist."""
    initialize_auth()
    username = _get_setting("SMART_HR_ADMIN_USERNAME").strip()
    password = _get_setting("SMART_HR_ADMIN_PASSWORD")
    with _connect() as connection:
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if user_count:
        return False
    if not username and not password:
        return False
    if not username or not password:
        raise AuthConfigurationError(
            "Set both SMART_HR_ADMIN_USERNAME and SMART_HR_ADMIN_PASSWORD "
            "to create the initial admin account."
        )
    try:
        create_user(username, password, role="admin")
    except ValueError as exc:
        raise AuthConfigurationError(
            f"Could not create the initial admin account: {exc}"
        ) from exc
    return True


def _normalize_username(username: str) -> str:
    normalized = username.strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,31}", normalized):
        raise ValueError(
            "Username must be 3-32 characters and contain only letters, "
            "numbers, periods, underscores, or hyphens."
        )
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    if len(password) > 256:
        raise ValueError("Password must be no more than 256 characters long.")


def create_user(username: str, password: str, role: str = "user") -> int:
    """Create a local account and return its ID."""
    normalized = _normalize_username(username)
    _validate_password(password)
    if role not in {"admin", "user"}:
        raise ValueError("Role must be 'admin' or 'user'.")

    password_hash = PASSWORD_HASHER.hash(password)
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized, password_hash, role, created_at),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username already exists.") from exc
    return cursor.lastrowid


def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials; return public user fields or None on invalid login."""
    normalized = username.strip().casefold()
    if len(normalized) > 128:
        normalized = normalized[:128]
    now = int(time.time())
    with _connect() as connection:
        connection.execute(
            "DELETE FROM login_attempts WHERE attempted_at < ?",
            (now - LOGIN_WINDOW_SECONDS,),
        )
        recent_failures = connection.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE username = ? AND attempted_at >= ?
            """,
            (normalized, now - LOGIN_WINDOW_SECONDS),
        ).fetchone()[0]
        if recent_failures >= MAX_LOGIN_ATTEMPTS:
            return None

        user = connection.execute(
            """
            SELECT id, username, password_hash, role, active, session_version
            FROM users WHERE username = ?
            """,
            (normalized,),
        ).fetchone()
        password_hash = user["password_hash"] if user else DUMMY_PASSWORD_HASH
        valid_password = False
        try:
            valid_password = PASSWORD_HASHER.verify(password_hash, password)
        except (VerifyMismatchError, InvalidHashError, TypeError):
            pass

        if not user or not user["active"] or not valid_password:
            connection.execute(
                "INSERT INTO login_attempts (username, attempted_at) VALUES (?, ?)",
                (normalized, now),
            )
            return None

        connection.execute(
            "DELETE FROM login_attempts WHERE username = ?",
            (normalized,),
        )
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "session_version": user["session_version"],
        }


def get_active_user(user_id: int) -> dict | None:
    """Return public details for an active account."""
    with _connect() as connection:
        user = connection.execute(
            "SELECT id, username, role, session_version "
            "FROM users WHERE id = ? AND active = 1",
            (user_id,),
        ).fetchone()
    return dict(user) if user else None


def list_users() -> list[dict]:
    """List account details without exposing password hashes."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, username, role, active, created_at FROM users ORDER BY username"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def reset_password(user_id: int, new_password: str) -> None:
    """Set a new password for an existing account."""
    _validate_password(new_password)
    password_hash = PASSWORD_HASHER.hash(new_password)
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE users SET password_hash = ?, "
            "session_version = session_version + 1 WHERE id = ?",
            (password_hash, user_id),
        )
        if not cursor.rowcount:
            raise ValueError("User account not found.")
        user = connection.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        connection.execute(
            "DELETE FROM login_attempts WHERE username = ?",
            (user["username"],),
        )


def set_user_active(user_id: int, active: bool, actor_id: int) -> None:
    """Enable or disable an account without allowing self-disable or lockout."""
    with _connect() as connection:
        target = connection.execute(
            "SELECT id, role, active FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not target:
            raise ValueError("User account not found.")
        if not active and user_id == actor_id:
            raise ValueError("You cannot disable your own account.")
        if not active and target["role"] == "admin" and target["active"]:
            active_admins = connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1"
            ).fetchone()[0]
            if active_admins <= 1:
                raise ValueError("The last active admin account cannot be disabled.")
        connection.execute(
            "UPDATE users SET active = ? WHERE id = ?",
            (int(active), user_id),
        )
