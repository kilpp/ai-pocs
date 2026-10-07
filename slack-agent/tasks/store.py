import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

from cryptography.fernet import Fernet

from tasks.models import TaskSpec

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    source_channel  TEXT NOT NULL,
    source_ts       TEXT NOT NULL,
    spec            TEXT,
    status          TEXT NOT NULL,
    assignee        TEXT,
    task_channel    TEXT,
    task_ts         TEXT,
    progress_ts     TEXT,
    pr_url          TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE (source_channel, source_ts)
);
CREATE TABLE IF NOT EXISTS github_tokens (
    slack_user_id   TEXT PRIMARY KEY,
    github_login    TEXT NOT NULL,
    token           BLOB NOT NULL,
    updated_at      REAL NOT NULL
);
"""

# Task lifecycle
STATUS_CLASSIFYING = "classifying"
STATUS_IGNORED = "ignored"
STATUS_OPEN = "open"
STATUS_IN_PROGRESS = "in_progress"
STATUS_MANUAL = "manual"  # picked up, but automation was blocked by validation
STATUS_AWAITING_RESTART = "awaiting_restart"
STATUS_RESTARTING = "restarting"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class TaskRecord:
    id: str
    source_channel: str
    source_ts: str
    spec: TaskSpec | None
    status: str
    assignee: str | None
    task_channel: str | None
    task_ts: str | None
    progress_ts: str | None
    pr_url: str | None


class TaskStore:
    """SQLite-backed store for tasks and per-user GitHub tokens.

    Tasks must survive restarts (someone may click "Pick up" hours later), so
    unlike the in-memory SessionStore this one persists to disk.
    """

    def __init__(self, db_path: str, encryption_key: str):
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._fernet = Fernet(encryption_key) if encryption_key else None

    # -- tasks -------------------------------------------------------------

    def create_pending(self, source_channel: str, source_ts: str) -> str | None:
        """Reserve a task for a source message.

        Returns the new task ID, or None if this message was already seen
        (Slack retries events, so this de-duplicates them).
        """
        task_id = uuid.uuid4().hex[:8]
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO tasks (id, source_channel, source_ts, status,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, source_channel, source_ts, STATUS_CLASSIFYING, now, now),
            )
        return task_id if cur.rowcount == 1 else None

    def update(self, task_id: str, **fields) -> None:
        if "spec" in fields and isinstance(fields["spec"], TaskSpec):
            fields["spec"] = json.dumps(fields["spec"].to_dict())
        fields["updated_at"] = time.time()
        columns = ", ".join(f"{name} = ?" for name in fields)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE tasks SET {columns} WHERE id = ?",
                (*fields.values(), task_id),
            )

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return TaskRecord(
            id=row["id"],
            source_channel=row["source_channel"],
            source_ts=row["source_ts"],
            spec=TaskSpec.from_dict(json.loads(row["spec"])) if row["spec"] else None,
            status=row["status"],
            assignee=row["assignee"],
            task_channel=row["task_channel"],
            task_ts=row["task_ts"],
            progress_ts=row["progress_ts"],
            pr_url=row["pr_url"],
        )

    def claim(self, task_id: str, user_id: str) -> bool:
        """Atomically assign an open task. Only the first clicker wins."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET status = ?, assignee = ?, updated_at = ?"
                " WHERE id = ? AND status = ?",
                (STATUS_IN_PROGRESS, user_id, time.time(), task_id, STATUS_OPEN),
            )
        return cur.rowcount == 1

    def transition(self, task_id: str, from_status: str, to_status: str) -> bool:
        """Compare-and-set the status, guarding against double clicks."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (to_status, time.time(), task_id, from_status),
            )
        return cur.rowcount == 1

    # -- GitHub tokens -----------------------------------------------------

    def save_github_token(self, slack_user_id: str, login: str, token: str) -> None:
        if self._fernet is None:
            raise RuntimeError("TOKEN_ENCRYPTION_KEY is not set")
        encrypted = self._fernet.encrypt(token.encode())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO github_tokens VALUES (?, ?, ?, ?)",
                (slack_user_id, login, encrypted, time.time()),
            )

    def get_github_token(self, slack_user_id: str) -> tuple[str, str] | None:
        """Return (github_login, token) for a Slack user, if linked."""
        if self._fernet is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT github_login, token FROM github_tokens WHERE slack_user_id = ?",
                (slack_user_id,),
            ).fetchone()
        if row is None:
            return None
        return row["github_login"], self._fernet.decrypt(row["token"]).decode()

    def delete_github_token(self, slack_user_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM github_tokens WHERE slack_user_id = ?", (slack_user_id,)
            )
