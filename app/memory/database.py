"""SQLite Persistent Memory Database for University AI Agent."""

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiosqlite

from app.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mastery_level REAL DEFAULT 0.0,
    notes TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learning_sessions (
    id TEXT PRIMARY KEY,
    session_type TEXT NOT NULL,
    title TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id TEXT PRIMARY KEY,
    topic_id TEXT,
    question TEXT NOT NULL,
    student_answer TEXT,
    correct_answer TEXT,
    is_correct INTEGER NOT NULL,
    score REAL DEFAULT 0.0,
    feedback TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS study_goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    target_date TEXT,
    status TEXT DEFAULT 'in_progress',
    progress_percent REAL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS academic_tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    subject_id TEXT,
    deadline TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mode TEXT DEFAULT 'general',
    citations TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON academic_tasks(status);
CREATE INDEX IF NOT EXISTS idx_topics_subject ON topics(subject_id);
"""


class DatabaseManager:
    """Async & sync manager for SQLite persistent memory."""

    def __init__(self, db_path: Optional[Path] = None, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.db_path = db_path or self.settings.SQLITE_PATH
        self._initialized = False

    def get_connection_sync(self) -> sqlite3.Connection:
        if not self._initialized and self.db_path.parent.exists():
            pass
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db_sync(self) -> None:
        """Initialize database tables synchronously on startup."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection_sync() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self._initialized = True

    async def init_db(self) -> None:
        """Initialize database tables asynchronously."""
        self.init_db_sync()

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager yielding an aiosqlite connection with row factory and foreign keys."""
        if not self._initialized:
            self.init_db_sync()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON;")
            yield conn

    # --- SUBJECTS & TOPICS ---

    async def add_subject(self, subject_id: str, name: str, description: str = "") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO subjects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (subject_id, name, description, now)
            )
            await conn.commit()
            return {"id": subject_id, "name": name, "description": description, "created_at": now}

    async def list_subjects(self) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM subjects ORDER BY name ASC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def add_or_update_topic(self, topic_id: str, subject_id: str, name: str, mastery: float = 0.0, notes: str = "") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO topics (id, subject_id, name, mastery_level, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    mastery_level = excluded.mastery_level,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (topic_id, subject_id, name, mastery, notes, now)
            )
            await conn.commit()
            return {"id": topic_id, "subject_id": subject_id, "name": name, "mastery_level": mastery, "notes": notes, "updated_at": now}

    async def list_topics(self, subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            if subject_id:
                cursor = await conn.execute("SELECT * FROM topics WHERE subject_id = ? ORDER BY name ASC", (subject_id,))
            else:
                cursor = await conn.execute("SELECT * FROM topics ORDER BY name ASC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # --- SESSIONS & CONVERSATION ---

    async def create_session(self, session_id: str, session_type: str = "tutor", title: str = "Academic Session") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO learning_sessions (id, session_type, title, started_at) VALUES (?, ?, ?, ?)",
                (session_id, session_type, title, now)
            )
            await conn.commit()
            return {"id": session_id, "session_type": session_type, "title": title, "started_at": now}

    async def save_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        mode: str = "general",
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations or [])
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, mode, citations, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, role, content, mode, citations_json, now)
            )
            await conn.commit()
            return {
                "id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "mode": mode,
                "citations": citations or [],
                "timestamp": now
            }

    async def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            )
            rows = await cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                if item.get("citations"):
                    try:
                        item["citations"] = json.loads(item["citations"])
                    except Exception:
                        item["citations"] = []
                result.append(item)
            return result

    # --- QUIZZES & LEARNING PROGRESS ---

    async def record_quiz_result(
        self,
        result_id: str,
        topic_id: Optional[str],
        question: str,
        student_answer: str,
        correct_answer: str,
        is_correct: bool,
        score: float,
        feedback: str
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO quiz_results (id, topic_id, question, student_answer, correct_answer, is_correct, score, feedback, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (result_id, topic_id, question, student_answer, correct_answer, 1 if is_correct else 0, score, feedback, now)
            )
            # Update topic mastery level dynamically
            if topic_id:
                cursor = await conn.execute(
                    "SELECT AVG(score) as avg_score FROM quiz_results WHERE topic_id = ?",
                    (topic_id,)
                )
                row = await cursor.fetchone()
                if row and row["avg_score"] is not None:
                    await conn.execute(
                        "UPDATE topics SET mastery_level = ?, updated_at = ? WHERE id = ?",
                        (float(row["avg_score"]), now, topic_id)
                    )
            await conn.commit()
            return {
                "id": result_id,
                "topic_id": topic_id,
                "question": question,
                "is_correct": is_correct,
                "score": score,
                "feedback": feedback,
                "created_at": now
            }

    async def get_learning_progress(self) -> Dict[str, Any]:
        """Aggregate student learning progress across topics and quizzes."""
        async with self.get_connection() as conn:
            c1 = await conn.execute("SELECT COUNT(*) as total_quizzes, AVG(score) as avg_score, SUM(is_correct) as total_correct FROM quiz_results")
            q_stats = await c1.fetchone()

            c2 = await conn.execute("SELECT * FROM topics ORDER BY mastery_level DESC")
            topics = [dict(r) for r in await c2.fetchall()]

            return {
                "total_quizzes_taken": q_stats["total_quizzes"] if q_stats else 0,
                "average_score": round(float(q_stats["avg_score"] or 0.0), 2) if q_stats else 0.0,
                "total_correct": q_stats["total_correct"] if q_stats else 0,
                "topic_breakdown": topics
            }

    # --- TASKS & GOALS ---

    async def add_task(
        self,
        task_id: str,
        title: str,
        description: str = "",
        subject_id: Optional[str] = None,
        deadline: Optional[str] = None,
        priority: str = "medium"
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO academic_tasks (id, title, description, subject_id, deadline, priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (task_id, title, description, subject_id, deadline, priority, now)
            )
            await conn.commit()
            return {
                "id": task_id,
                "title": title,
                "description": description,
                "subject_id": subject_id,
                "deadline": deadline,
                "priority": priority,
                "status": "pending",
                "created_at": now
            }

    async def update_task_status(self, task_id: str, status: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status == "completed" else None
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "UPDATE academic_tasks SET status = ?, completed_at = ? WHERE id = ?",
                (status, completed_at, task_id)
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            if status:
                cursor = await conn.execute("SELECT * FROM academic_tasks WHERE status = ? ORDER BY deadline ASC", (status,))
            else:
                cursor = await conn.execute("SELECT * FROM academic_tasks ORDER BY deadline ASC")
            return [dict(r) for r in await cursor.fetchall()]

    # --- USER PREFERENCES ---

    async def set_preference(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, now)
            )
            await conn.commit()

    async def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row["value"] if row else default


db = DatabaseManager()
