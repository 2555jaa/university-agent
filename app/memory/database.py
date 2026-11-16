"""SQLite Persistent Memory Database for University AI Agent with Migrations and Hardened Pragmas."""

import asyncio
import hashlib
import json
import logging
import re
import shutil
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import aiosqlite

from app.config import settings

logger = logging.getLogger("university_agent.memory.database")


@dataclass
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _migration_001_initial_schema(conn: sqlite3.Connection) -> None:
    """Baseline initial schema creation."""
    conn.executescript("""
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
        last_reviewed TEXT,
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
        metadata TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON academic_tasks(status);
    CREATE INDEX IF NOT EXISTS idx_topics_subject ON topics(subject_id);
    """)


def _migration_002_constraints_and_indexes(conn: sqlite3.Connection) -> None:
    """Enforce constraints (checks, uniqueness, cascades) and create performance indexes."""
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("BEGIN TRANSACTION;")

    # 1. subjects: add code column and case-insensitive unique indexes
    cols = [r[1] for r in conn.execute("PRAGMA table_info(subjects)").fetchall()]
    if "code" not in cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN code TEXT;")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_subjects_id_lower ON subjects(LOWER(id));")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_subjects_name_lower ON subjects(LOWER(name));")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_subjects_code_lower ON subjects(LOWER(code)) WHERE code IS NOT NULL;")

    # 2. topics: mastery check, last_reviewed column, and unique (subject_id, name)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(topics)").fetchall()]
    has_last_rev = "last_reviewed" in cols
    conn.execute("""
    CREATE TABLE topics_new (
        id TEXT PRIMARY KEY,
        subject_id TEXT NOT NULL,
        name TEXT NOT NULL,
        mastery_level REAL DEFAULT 0.0 CHECK(mastery_level >= 0.0 AND mastery_level <= 1.0),
        notes TEXT,
        last_reviewed TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
    );
    """)
    if has_last_rev:
        conn.execute("INSERT INTO topics_new SELECT id, subject_id, name, mastery_level, notes, last_reviewed, updated_at FROM topics;")
    else:
        conn.execute("INSERT INTO topics_new SELECT id, subject_id, name, mastery_level, notes, NULL AS last_reviewed, updated_at FROM topics;")
    conn.execute("DROP TABLE topics;")
    conn.execute("ALTER TABLE topics_new RENAME TO topics;")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_subject_name_lower ON topics(subject_id, LOWER(name));")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_subject ON topics(subject_id);")

    # 3. learning_sessions: session_type check and indexes
    cols = [r[1] for r in conn.execute("PRAGMA table_info(learning_sessions)").fetchall()]
    if "subject_id" not in cols:
        conn.execute("ALTER TABLE learning_sessions ADD COLUMN subject_id TEXT;")
    if "updated_at" not in cols:
        conn.execute("ALTER TABLE learning_sessions ADD COLUMN updated_at TEXT;")
    conn.execute("""
    CREATE TABLE learning_sessions_new (
        id TEXT PRIMARY KEY,
        session_type TEXT NOT NULL CHECK(session_type IN ('tutor', 'research', 'assignment', 'assistant', 'general')),
        title TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        summary TEXT,
        subject_id TEXT,
        updated_at TEXT,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
    );
    """)
    conn.execute("INSERT INTO learning_sessions_new SELECT id, session_type, title, started_at, ended_at, summary, subject_id, updated_at FROM learning_sessions;")
    conn.execute("DROP TABLE learning_sessions;")
    conn.execute("ALTER TABLE learning_sessions_new RENAME TO learning_sessions;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_subject_updated ON learning_sessions(subject_id, updated_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON learning_sessions(started_at);")

    # 4. messages: role domain check, metadata column, and composite index (session_id, timestamp)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
    has_meta = "metadata" in cols
    conn.execute("""
    CREATE TABLE messages_new (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        mode TEXT DEFAULT 'general',
        citations TEXT,
        metadata TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE
    );
    """)
    if has_meta:
        conn.execute("INSERT INTO messages_new SELECT id, session_id, role, content, mode, citations, metadata, timestamp FROM messages;")
    else:
        conn.execute("INSERT INTO messages_new SELECT id, session_id, role, content, mode, citations, NULL AS metadata, timestamp FROM messages;")
    conn.execute("DROP TABLE messages;")
    conn.execute("ALTER TABLE messages_new RENAME TO messages;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp);")

    # 5. quiz_results: score/is_correct check, topic FK, subject_id, and indexes
    cols = [r[1] for r in conn.execute("PRAGMA table_info(quiz_results)").fetchall()]
    if "subject_id" not in cols:
        conn.execute("ALTER TABLE quiz_results ADD COLUMN subject_id TEXT;")
    conn.execute("""
    CREATE TABLE quiz_results_new (
        id TEXT PRIMARY KEY,
        topic_id TEXT,
        subject_id TEXT,
        question TEXT NOT NULL,
        student_answer TEXT,
        correct_answer TEXT,
        is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
        score REAL DEFAULT 0.0 CHECK(score >= 0.0 AND score <= 1.0),
        feedback TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE SET NULL,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
    );
    """)
    conn.execute("INSERT INTO quiz_results_new SELECT id, topic_id, subject_id, question, student_answer, correct_answer, is_correct, score, feedback, created_at FROM quiz_results;")
    conn.execute("DROP TABLE quiz_results;")
    conn.execute("ALTER TABLE quiz_results_new RENAME TO quiz_results;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_subject_topic ON quiz_results(subject_id, topic_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_topic_created ON quiz_results(topic_id, created_at);")

    # 6. academic_tasks: priority and status checks, subject FK, composite index (status, deadline)
    conn.execute("""
    CREATE TABLE academic_tasks_new (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        subject_id TEXT,
        deadline TEXT,
        priority TEXT DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high')),
        status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'cancelled')),
        created_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
    );
    """)
    conn.execute("INSERT INTO academic_tasks_new SELECT id, title, description, subject_id, deadline, priority, status, created_at, completed_at FROM academic_tasks;")
    conn.execute("DROP TABLE academic_tasks;")
    conn.execute("ALTER TABLE academic_tasks_new RENAME TO academic_tasks;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_deadline ON academic_tasks(status, deadline);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_subject ON academic_tasks(subject_id);")

    # 7. study_goals: status and progress checks
    cols = [r[1] for r in conn.execute("PRAGMA table_info(study_goals)").fetchall()]
    if "subject_id" not in cols:
        conn.execute("ALTER TABLE study_goals ADD COLUMN subject_id TEXT;")
    conn.execute("""
    CREATE TABLE study_goals_new (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        target_date TEXT,
        status TEXT DEFAULT 'in_progress' CHECK(status IN ('in_progress', 'completed', 'archived', 'abandoned')),
        progress_percent REAL DEFAULT 0.0 CHECK(progress_percent >= 0.0 AND progress_percent <= 100.0),
        created_at TEXT NOT NULL,
        subject_id TEXT,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
    );
    """)
    conn.execute("INSERT INTO study_goals_new SELECT id, title, description, target_date, status, progress_percent, created_at, subject_id FROM study_goals;")
    conn.execute("DROP TABLE study_goals;")
    conn.execute("ALTER TABLE study_goals_new RENAME TO study_goals;")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_study_goals_status ON study_goals(status);")

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON;")


def _migration_003_documents_catalog(conn: sqlite3.Connection) -> None:
    """Create documents metadata catalog table with unique indexes and constraints."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        file_type TEXT NOT NULL CHECK(file_type IN ('pdf', 'docx', 'txt', 'md')),
        file_size_bytes INTEGER NOT NULL CHECK(file_size_bytes >= 0),
        subject_id TEXT REFERENCES subjects(id) ON DELETE SET NULL,
        title TEXT,
        page_count INTEGER NOT NULL DEFAULT 1 CHECK(page_count >= 1),
        chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
        status TEXT NOT NULL DEFAULT 'indexed' CHECK(status IN ('pending', 'processing', 'indexed', 'failed')),
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
    CREATE INDEX IF NOT EXISTS idx_documents_subject ON documents(subject_id);
    CREATE INDEX IF NOT EXISTS idx_documents_status_updated ON documents(status, updated_at);
    CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
    """)


MIGRATIONS: List[Migration] = [
    Migration(version=1, name="001_initial_schema", apply=_migration_001_initial_schema),
    Migration(version=2, name="002_constraints_and_indexes", apply=_migration_002_constraints_and_indexes),
    Migration(version=3, name="003_documents_catalog", apply=_migration_003_documents_catalog),
]


class DatabaseManager:
    """Async & sync manager for SQLite persistent memory with migration support and hardened pragmas."""

    def __init__(self, db_path: Optional[Path] = None, settings_obj: Optional[Any] = None):
        self.settings = settings_obj or settings
        self.db_path = db_path or self.settings.SQLITE_PATH
        self._initialized = False

    def _configure_connection_sync(self, conn: sqlite3.Connection) -> None:
        """Apply production pragmas consistently on synchronous connections."""
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")

    async def _configure_connection_async(self, conn: aiosqlite.Connection) -> None:
        """Apply production pragmas consistently on asynchronous connections."""
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON;")
        if str(self.db_path) != ":memory:":
            await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA synchronous = NORMAL;")
        await conn.execute("PRAGMA busy_timeout = 5000;")

    def get_connection_sync(self) -> sqlite3.Connection:
        """Create and configure a synchronous SQLite connection with hardened pragmas."""
        conn = sqlite3.connect(self.db_path)
        self._configure_connection_sync(conn)
        return conn

    def get_schema_version_sync(self) -> int:
        """Return the current schema migration version synchronously."""
        if not self.db_path.exists():
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
                if not cursor.fetchone():
                    return 0
                cursor = conn.execute("SELECT MAX(version) FROM schema_migrations")
                row = cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    async def get_schema_version(self) -> int:
        """Return the current schema migration version asynchronously."""
        return self.get_schema_version_sync()

    def run_migrations_sync(self) -> int:
        """Run pending schema migrations sequentially in a deterministic transaction."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection_sync() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

            cursor = conn.execute("SELECT version FROM schema_migrations")
            applied_versions = {row[0] for row in cursor.fetchall()}

            for migration in MIGRATIONS:
                if migration.version not in applied_versions:
                    logger.info(f"Applying SQLite migration {migration.version:03d}_{migration.name}...")
                    migration.apply(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                        (migration.version, migration.name)
                    )
                    conn.commit()

        self._initialized = True
        return self.get_schema_version_sync()

    def init_db_sync(self) -> None:
        """Initialize database tables synchronously on startup via migrations."""
        self.run_migrations_sync()

    async def init_db(self) -> None:
        """Initialize database tables asynchronously."""
        self.init_db_sync()

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager yielding an aiosqlite connection with hardened pragmas."""
        if not self._initialized:
            self.init_db_sync()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection_async(conn)
            yield conn

    @asynccontextmanager
    async def transaction(self):
        """Async context manager for atomic transactions with automatic rollback on error."""
        if not self._initialized:
            self.init_db_sync()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._configure_connection_async(conn)
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def check_health(self) -> Dict[str, Any]:
        """Check database connectivity, integrity, and schema version."""
        try:
            async with self.get_connection() as conn:
                async with conn.execute("SELECT 1") as cursor:
                    await cursor.fetchone()
            schema_ver = self.get_schema_version_sync()
            integrity = self.verify_integrity_sync()
            return {
                "status": "ok" if integrity["ok"] else "degraded",
                "path": str(self.db_path),
                "schema_version": schema_ver,
                "wal_mode": True if str(self.db_path) != ":memory:" else False,
                "integrity": integrity["integrity_check"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "path": str(self.db_path)}

    # --- BACKUP, INTEGRITY & RECOVERY ---

    def verify_integrity_sync(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Verify SQLite database integrity and foreign key constraints."""
        target_path = Path(path) if path else self.db_path
        if str(target_path) != ":memory:" and not target_path.exists():
            return {
                "ok": False,
                "error": f"Database file not found: {target_path}",
                "integrity_check": "file_not_found",
                "foreign_key_violations": [],
                "schema_version": 0
            }

        try:
            with sqlite3.connect(target_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 5000;")

                cursor = conn.execute("PRAGMA integrity_check;")
                rows = cursor.fetchall()
                raw_results = [r[0] if not isinstance(r, (str, bytes)) else r for r in rows]
                integrity_ok = len(raw_results) == 1 and raw_results[0] == "ok"
                integrity_str = raw_results[0] if len(raw_results) == 1 else "; ".join(str(x) for x in raw_results)

                cursor = conn.execute("PRAGMA foreign_key_check;")
                fk_rows = cursor.fetchall()
                fk_violations = [list(r) for r in fk_rows]

                schema_version = 0
                tbl_cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
                if tbl_cursor.fetchone():
                    v_cursor = conn.execute("SELECT MAX(version) FROM schema_migrations")
                    v_row = v_cursor.fetchone()
                    schema_version = int(v_row[0]) if v_row and v_row[0] is not None else 0

                ok = integrity_ok and len(fk_violations) == 0
                res: Dict[str, Any] = {
                    "ok": ok,
                    "integrity_check": integrity_str,
                    "foreign_key_violations": fk_violations,
                    "schema_version": schema_version
                }
                if not ok and integrity_ok and fk_violations:
                    res["error"] = f"Foreign key check failed with {len(fk_violations)} violations."
                elif not ok and not integrity_ok:
                    res["error"] = f"Integrity check failed: {integrity_str}"
                return res

        except sqlite3.DatabaseError as e:
            logger.warning(f"Database integrity check failed with DatabaseError on {target_path}: {e}")
            return {
                "ok": False,
                "error": str(e),
                "integrity_check": str(e),
                "foreign_key_violations": [],
                "schema_version": 0
            }
        except Exception as e:
            logger.warning(f"Database integrity check failed with Exception on {target_path}: {e}")
            return {
                "ok": False,
                "error": str(e),
                "integrity_check": str(e),
                "foreign_key_violations": [],
                "schema_version": 0
            }

    async def verify_integrity(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Verify database integrity asynchronously."""
        return await asyncio.to_thread(self.verify_integrity_sync, path)

    def create_backup_sync(self, backup_dir: Optional[Path] = None, label: str = "manual") -> Dict[str, Any]:
        """Create a hot-safe SQLite online backup with companion metadata and integrity verification."""
        if str(self.db_path) == ":memory:":
            raise ValueError("Cannot create file backup of in-memory database.")
        if not self.db_path.exists():
            self.init_db_sync()

        target_dir = Path(backup_dir) if backup_dir else (self.db_path.parent / "backups")
        target_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        cleaned_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).strip() or "manual"
        backup_filename = f"university_agent_{cleaned_label}_{timestamp}.db"
        dest_path = target_dir / backup_filename
        meta_path = target_dir / f"{backup_filename}.meta.json"

        # Hot online backup via SQLite native backup API
        source_conn = sqlite3.connect(self.db_path)
        dest_conn = sqlite3.connect(dest_path)
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            source_conn.close()

        # Post-backup integrity verification
        integrity_report = self.verify_integrity_sync(dest_path)
        if not integrity_report["ok"]:
            if dest_path.exists():
                dest_path.unlink()
            raise RuntimeError(f"Backup verification failed: {integrity_report}")

        # Compute SHA-256 and size
        h = hashlib.sha256()
        with open(dest_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha256_hex = h.hexdigest()
        size_bytes = dest_path.stat().st_size

        # Table row counts
        table_counts: Dict[str, int] = {}
        with sqlite3.connect(dest_path) as b_conn:
            b_conn.row_factory = sqlite3.Row
            cur = b_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [r["name"] for r in cur.fetchall()]
            for t in sorted(tables):
                try:
                    c_cur = b_conn.execute(f"SELECT COUNT(*) FROM {t}")
                    table_counts[t] = c_cur.fetchone()[0]
                except Exception:
                    pass

        schema_version = integrity_report.get("schema_version", self.get_schema_version_sync())
        manifest = {
            "backup_file": backup_filename,
            "backup_path": str(dest_path),
            "meta_path": str(meta_path),
            "created_at": now.isoformat(),
            "label": cleaned_label,
            "sha256": sha256_hex,
            "size_bytes": size_bytes,
            "schema_version": schema_version,
            "table_counts": table_counts,
            "integrity": integrity_report["integrity_check"]
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created verified backup: {backup_filename} ({size_bytes} bytes, sha256: {sha256_hex[:8]})")
        return manifest

    async def create_backup(self, backup_dir: Optional[Path] = None, label: str = "manual") -> Dict[str, Any]:
        """Create a hot backup asynchronously."""
        return await asyncio.to_thread(self.create_backup_sync, backup_dir, label)

    def restore_backup_sync(self, backup_path: Path) -> Dict[str, Any]:
        """Atomically restore active database from a verified backup file with safety rollback protection."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file does not exist: {backup_path}")

        # 1. Validate manifest if companion .meta.json exists
        meta_path = backup_path.with_name(f"{backup_path.name}.meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                expected_sha = meta.get("sha256")
                if expected_sha:
                    h = hashlib.sha256()
                    with open(backup_path, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    actual_sha = h.hexdigest()
                    if actual_sha != expected_sha:
                        raise ValueError(f"Backup SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Failed to validate backup metadata manifest: {e}")

        # 2. Validate backup integrity before touching active database
        backup_integrity = self.verify_integrity_sync(backup_path)
        if not backup_integrity["ok"]:
            err_msg = backup_integrity.get("error") or backup_integrity.get("integrity_check", "Corrupt database")
            raise ValueError(f"Backup file failed integrity verification: {err_msg}")

        # 3. Take safety snapshot of active database if it exists
        pre_restore_snapshot = None
        if self.db_path.exists():
            snapshot_dir = self.db_path.parent / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            pre_restore_snapshot = snapshot_dir / f"{self.db_path.stem}_pre_restore_{ts}.db"
            try:
                shutil.copy2(self.db_path, pre_restore_snapshot)
            except Exception as e:
                logger.warning(f"Failed to create pre-restore snapshot: {e}")

        # 4. Remove stale WAL sidecars before replacing active DB file
        wal_file = self.db_path.with_name(f"{self.db_path.name}-wal")
        shm_file = self.db_path.with_name(f"{self.db_path.name}-shm")
        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()

        # 5. Atomic restore of active DB file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, self.db_path)

        # 6. Ensure schema is initialized and migrations up to date
        self._initialized = False
        self.init_db_sync()

        # 7. Post-restore integrity verification
        post_integrity = self.verify_integrity_sync()
        if not post_integrity["ok"]:
            # Rollback to pre-restore snapshot if available
            if pre_restore_snapshot and pre_restore_snapshot.exists():
                shutil.copy2(pre_restore_snapshot, self.db_path)
                self.init_db_sync()
            raise RuntimeError(f"Restored database failed integrity verification: {post_integrity}")

        logger.info(f"Successfully restored database from {backup_path}")
        return {
            "ok": True,
            "restored_from": str(backup_path),
            "pre_restore_snapshot": str(pre_restore_snapshot) if pre_restore_snapshot else None,
            "schema_version": self.get_schema_version_sync(),
            "integrity": post_integrity
        }

    async def restore_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Restore database from backup asynchronously."""
        return await asyncio.to_thread(self.restore_backup_sync, backup_path)

    def recover_or_restore_sync(self, fallback_backup_path: Optional[Path] = None) -> Dict[str, Any]:
        """Attempt to recover from corrupted state via checkpointing or restoring fallback backup."""
        if not self.db_path.exists():
            if fallback_backup_path and Path(fallback_backup_path).exists():
                restore_res = self.restore_backup_sync(Path(fallback_backup_path))
                return {
                    "recovered": True,
                    "strategy": "restored_from_backup",
                    "details": restore_res
                }
            return {
                "recovered": False,
                "strategy": "none",
                "error": f"Database file does not exist: {self.db_path}"
            }

        # Step 1: Attempt WAL checkpoint recovery (useful for interrupted writes)
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._configure_connection_sync(conn)
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception as e:
            logger.warning(f"WAL checkpoint recovery attempted with warning/error: {e}")

        # Step 2: Check integrity of active db
        integrity = self.verify_integrity_sync()
        if integrity["ok"]:
            return {
                "recovered": True,
                "strategy": "checkpoint",
                "integrity": integrity
            }

        # Step 3: If still corrupted, attempt restore from fallback backup
        if fallback_backup_path and Path(fallback_backup_path).exists():
            logger.warning(f"Active database corrupted ({integrity.get('error')}). Restoring from fallback: {fallback_backup_path}")
            try:
                restore_res = self.restore_backup_sync(Path(fallback_backup_path))
                return {
                    "recovered": True,
                    "strategy": "restored_from_backup",
                    "details": restore_res
                }
            except Exception as e:
                return {
                    "recovered": False,
                    "strategy": "restore_failed",
                    "error": str(e),
                    "active_integrity": integrity
                }

        return {
            "recovered": False,
            "strategy": "corrupted_no_backup",
            "active_integrity": integrity
        }

    async def recover_or_restore(self, fallback_backup_path: Optional[Path] = None) -> Dict[str, Any]:
        """Recover or restore database asynchronously."""
        return await asyncio.to_thread(self.recover_or_restore_sync, fallback_backup_path)

    # --- SUBJECTS & TOPICS ---

    async def add_subject(
        self,
        subject_id: str,
        name: str,
        description: str = "",
        code: Optional[str] = None
    ) -> Dict[str, Any]:
        cleaned_id = subject_id.strip()
        cleaned_name = name.strip()
        if not cleaned_id:
            raise ValueError("Subject ID cannot be empty.")
        if not cleaned_name:
            raise ValueError("Subject name cannot be empty.")

        cleaned_code = (code.strip() if code else cleaned_id).upper()
        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            # Check for existing subject (idempotent lookup by id or name)
            cursor = await conn.execute(
                "SELECT id, name, description, code, created_at FROM subjects WHERE LOWER(id) = LOWER(?) OR LOWER(name) = LOWER(?)",
                (cleaned_id, cleaned_name)
            )
            existing = await cursor.fetchone()
            if existing:
                return dict(existing)

            await conn.execute(
                """
                INSERT INTO subjects (id, name, description, created_at, code)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cleaned_id, cleaned_name, description.strip(), now, cleaned_code)
            )
            return {"id": cleaned_id, "name": cleaned_name, "description": description.strip(), "code": cleaned_code, "created_at": now}

    async def delete_subject(self, subject_id: str) -> bool:
        cleaned_id = subject_id.strip()
        if not cleaned_id:
            raise ValueError("Subject ID cannot be empty.")
        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM subjects WHERE id = ?", (cleaned_id,))
            return cursor.rowcount > 0

    async def list_subjects(self) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM subjects ORDER BY name ASC")
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def add_or_update_topic(
        self,
        topic_id: str,
        subject_id: str,
        name: str,
        mastery: float = 0.0,
        notes: str = ""
    ) -> Dict[str, Any]:
        cleaned_id = topic_id.strip() if topic_id else ""
        cleaned_sub = subject_id.strip() if subject_id else ""
        cleaned_name = name.strip() if name else ""
        if not cleaned_id:
            raise ValueError("Topic ID cannot be empty.")
        if not cleaned_sub:
            raise ValueError("Subject ID cannot be empty.")
        if not cleaned_name:
            raise ValueError("Topic name cannot be empty.")
        if not (0.0 <= mastery <= 1.0):
            raise ValueError(f"Mastery level must be between 0.0 and 1.0, got {mastery}")

        now = datetime.now(timezone.utc).isoformat()
        async with self.transaction() as conn:
            # Ensure topic strictly belongs to a valid existing subject
            sub_cur = await conn.execute(
                "SELECT id FROM subjects WHERE id = ? OR LOWER(id) = LOWER(?)",
                (cleaned_sub, cleaned_sub)
            )
            subj = await sub_cur.fetchone()
            if not subj:
                raise ValueError(f"Subject '{cleaned_sub}' does not exist.")

            resolved_sub_id = subj[0]
            await conn.execute(
                """
                INSERT INTO topics (id, subject_id, name, mastery_level, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    subject_id = excluded.subject_id,
                    name = excluded.name,
                    mastery_level = excluded.mastery_level,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (cleaned_id, resolved_sub_id, cleaned_name, mastery, notes, now)
            )
            return {
                "id": cleaned_id,
                "subject_id": resolved_sub_id,
                "name": cleaned_name,
                "mastery_level": mastery,
                "notes": notes,
                "updated_at": now
            }

    async def get_topic_by_name(self, subject_id: str, topic_name: str) -> Optional[Dict[str, Any]]:
        """Fetch a topic by its name under a specific subject_id."""
        cleaned_sub = subject_id.strip() if subject_id else ""
        cleaned_name = topic_name.strip() if topic_name else ""
        if not cleaned_sub:
            raise ValueError("Subject ID cannot be empty.")
        if not cleaned_name:
            raise ValueError("Topic name cannot be empty.")

        async with self.get_connection() as conn:
            # Check subject exists
            sub_cur = await conn.execute(
                "SELECT id FROM subjects WHERE id = ? OR LOWER(id) = LOWER(?)",
                (cleaned_sub, cleaned_sub)
            )
            subj = await sub_cur.fetchone()
            if not subj:
                raise ValueError(f"Subject '{cleaned_sub}' does not exist.")

            cursor = await conn.execute(
                "SELECT * FROM topics WHERE subject_id = ? AND LOWER(name) = LOWER(?)",
                (subj[0], cleaned_name)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_topics(self, subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List topics, optionally filtered by subject_id."""
        if subject_id is not None:
            cleaned_sub = subject_id.strip()
            if not cleaned_sub:
                raise ValueError("Subject ID cannot be empty.")
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM topics WHERE subject_id = ? OR LOWER(subject_id) = LOWER(?) ORDER BY name ASC",
                    (cleaned_sub, cleaned_sub)
                )
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
        else:
            async with self.get_connection() as conn:
                cursor = await conn.execute("SELECT * FROM topics ORDER BY name ASC")
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    # --- SESSIONS & CONVERSATION ---

    async def create_session(
        self,
        session_id: str,
        session_type: str = "tutor",
        title: str = "Academic Session",
        subject_id: Optional[str] = None
    ) -> Dict[str, Any]:
        cleaned_id = session_id.strip()
        cleaned_type = session_type.strip().lower()
        if not cleaned_id:
            raise ValueError("Session ID cannot be empty.")
        if cleaned_type not in ("tutor", "research", "assignment", "assistant", "general"):
            raise ValueError(f"Invalid session_type '{session_type}'.")

        now = datetime.now(timezone.utc).isoformat()
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO learning_sessions (id, session_type, title, started_at, subject_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_type = excluded.session_type,
                    title = excluded.title,
                    subject_id = excluded.subject_id,
                    updated_at = excluded.updated_at
                """,
                (cleaned_id, cleaned_type, title.strip(), now, subject_id.strip() if subject_id else None, now)
            )
            return {"id": cleaned_id, "session_type": cleaned_type, "title": title.strip(), "started_at": now, "subject_id": subject_id}

    async def get_recent_sessions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent learning sessions ordered by latest activity."""
        safe_limit = max(1, limit)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, session_type, title, started_at, ended_at, summary, subject_id, updated_at
                FROM learning_sessions
                ORDER BY COALESCE(updated_at, started_at) DESC
                LIMIT ?
                """,
                (safe_limit,)
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def _has_messages_metadata_column(self, conn: Any) -> bool:
        cursor = await conn.execute("PRAGMA table_info(messages)")
        rows = await cursor.fetchall()
        cols = [r[1] for r in rows]
        return "metadata" in cols

    async def save_message(
        self,
        arg1: Optional[str] = None,
        arg2: Optional[str] = None,
        arg3: Optional[str] = None,
        arg4: Optional[Any] = None,
        mode: str = "general",
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        content: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Save a message with session association, mode, citations, and rich JSON metadata.

        Supports both standard signatures:
        1. save_message(session_id, role, content, metadata=None)
        2. save_message(message_id, session_id, role, content, mode, citations, metadata)
        3. keyword arguments: save_message(session_id=..., role=..., content=..., metadata=...)
        """
        import uuid as _uuid

        if session_id is not None:
            cleaned_id = str(message_id or arg1 or _uuid.uuid4()).strip()
            cleaned_sess = str(session_id).strip()
            cleaned_role = str(role or arg2 or "").strip().lower()
            cleaned_content = str(content if content is not None else (arg3 if arg3 is not None else "")).strip()
        elif arg2 is not None and str(arg2).strip().lower() in ("user", "assistant", "system"):
            # Signature: save_message(session_id, role, content, metadata=None)
            cleaned_id = str(message_id or _uuid.uuid4()).strip()
            cleaned_sess = str(arg1 or "").strip()
            cleaned_role = str(arg2).strip().lower()
            cleaned_content = str(arg3 if arg3 is not None else "").strip()
            if isinstance(arg4, dict) and metadata is None:
                metadata = arg4
        else:
            # Signature: save_message(message_id, session_id, role, content, ...)
            cleaned_id = str(arg1 or message_id or _uuid.uuid4()).strip()
            cleaned_sess = str(arg2 or "").strip()
            cleaned_role = str(arg3 or role or "").strip().lower()
            cleaned_content = str(arg4 if arg4 is not None else (content or "")).strip()
            cleaned_sess = (arg2 or "").strip()
            cleaned_role = (arg3 or "").strip().lower()
            cleaned_content = str(arg4 or "").strip()

        if not cleaned_id:
            raise ValueError("Message ID cannot be empty.")
        if not cleaned_sess:
            raise ValueError("Session ID cannot be empty.")
        if not cleaned_content:
            raise ValueError("Message content cannot be empty.")
        if cleaned_role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid message role '{cleaned_role}'.")

        # Synchronize citations and metadata
        if metadata is not None and not citations:
            if "used_sources" in metadata and isinstance(metadata["used_sources"], list):
                citations = metadata["used_sources"]
            elif "citations" in metadata and isinstance(metadata["citations"], list):
                citations = metadata["citations"]
        elif citations is not None and metadata is None:
            metadata = {"used_sources": citations}
        elif citations is not None and metadata is not None:
            if "used_sources" not in metadata:
                metadata["used_sources"] = citations

        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations or [])
        metadata_json = json.dumps(metadata or {})

        async with self.transaction() as conn:
            has_meta = await self._has_messages_metadata_column(conn)
            if not has_meta:
                try:
                    await conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT;")
                    has_meta = True
                except Exception:
                    has_meta = False

            if has_meta:
                await conn.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, mode, citations, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        role = excluded.role,
                        content = excluded.content,
                        mode = excluded.mode,
                        citations = excluded.citations,
                        metadata = excluded.metadata,
                        timestamp = excluded.timestamp
                    """,
                    (cleaned_id, cleaned_sess, cleaned_role, cleaned_content, mode, citations_json, metadata_json, now)
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, mode, citations, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        role = excluded.role,
                        content = excluded.content,
                        mode = excluded.mode,
                        citations = excluded.citations,
                        timestamp = excluded.timestamp
                    """,
                    (cleaned_id, cleaned_sess, cleaned_role, cleaned_content, mode, citations_json, now)
                )

            # Update session's updated_at timestamp
            await conn.execute(
                "UPDATE learning_sessions SET updated_at = ? WHERE id = ?",
                (now, cleaned_sess)
            )

            return {
                "id": cleaned_id,
                "session_id": cleaned_sess,
                "role": cleaned_role,
                "content": cleaned_content,
                "mode": mode,
                "citations": citations or [],
                "metadata": metadata or {},
                "timestamp": now
            }

    async def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id.strip(), limit)
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
                else:
                    item["citations"] = []

                if item.get("metadata"):
                    try:
                        item["metadata"] = json.loads(item["metadata"])
                    except Exception:
                        item["metadata"] = {}
                else:
                    item["metadata"] = {"used_sources": item["citations"]} if item["citations"] else {}

                result.append(item)
            return result

    # --- QUIZZES & LEARNING PROGRESS ---

    async def _has_topics_last_reviewed_column(self, conn: Any) -> bool:
        cursor = await conn.execute("PRAGMA table_info(topics)")
        rows = await cursor.fetchall()
        cols = [r[1] for r in rows]
        return "last_reviewed" in cols

    async def record_quiz_result(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Record a quiz evaluation result and update topic mastery via EMA.

        Supports both standard signatures:
        1. record_quiz_result(subject_id, topic_id, score, total_questions)
        2. record_quiz_result(result_id, topic_id, question, student_answer, correct_answer, is_correct, score, feedback, [subject_id])
        3. keyword arguments: record_quiz_result(subject_id=..., topic_id=..., score=..., ...)
        """
        import uuid as _uuid

        resolved_id = None
        resolved_subject_id = kwargs.get("subject_id")
        resolved_topic_id = kwargs.get("topic_id")
        resolved_q = kwargs.get("question")
        resolved_student_ans = kwargs.get("student_answer")
        resolved_correct_ans = kwargs.get("correct_answer")
        resolved_is_correct = kwargs.get("is_correct")
        resolved_score = kwargs.get("score")
        resolved_feedback = kwargs.get("feedback")
        resolved_total = kwargs.get("total_questions")

        # Parse positional arguments
        if len(args) == 4 and isinstance(args[2], (int, float)):
            # Pattern: (subject_id, topic_id, score, total_questions)
            resolved_subject_id = str(args[0]).strip() if args[0] else None
            resolved_topic_id = str(args[1]).strip() if args[1] else None
            resolved_score = float(args[2])
            resolved_total = int(args[3])
        elif len(args) >= 3 and isinstance(args[2], (int, float)):
            # Pattern: (subject_id, topic_id, score)
            resolved_subject_id = str(args[0]).strip() if args[0] else None
            resolved_topic_id = str(args[1]).strip() if args[1] else None
            resolved_score = float(args[2])
            resolved_total = int(args[3]) if len(args) > 3 else 1
        elif len(args) >= 7:
            # Pattern: (result_id, topic_id, question, student_answer, correct_answer, is_correct, score, feedback, [subject_id])
            resolved_id = str(args[0]).strip() if args[0] else None
            resolved_topic_id = str(args[1]).strip() if args[1] else None
            resolved_q = str(args[2]).strip()
            resolved_student_ans = str(args[3])
            resolved_correct_ans = str(args[4])
            resolved_is_correct = bool(args[5])
            resolved_score = float(args[6])
            if len(args) > 7:
                resolved_feedback = str(args[7])
            if len(args) > 8:
                resolved_subject_id = str(args[8]).strip() if args[8] else None
        elif len(args) == 1:
            resolved_id = str(args[0]).strip()

        # Defaults for missing fields
        if resolved_score is None:
            resolved_score = 0.0
        else:
            resolved_score = float(resolved_score)

        if not resolved_id:
            resolved_id = str(kwargs.get("result_id") or _uuid.uuid4()).strip()

        resolved_topic_id = str(resolved_topic_id).strip() if resolved_topic_id else None
        resolved_total = int(resolved_total or 1)

        if not resolved_q:
            resolved_q = f"Quiz on topic {resolved_topic_id or 'general'} ({resolved_total} questions)"

        if resolved_student_ans is None:
            resolved_student_ans = f"{int(resolved_score * resolved_total)}/{resolved_total} correct"

        if resolved_correct_ans is None:
            resolved_correct_ans = f"{resolved_total}/{resolved_total} correct"

        if resolved_is_correct is None:
            resolved_is_correct = (resolved_score >= 0.7)

        if not resolved_feedback:
            resolved_feedback = f"Score: {round(resolved_score * 100, 1)}%"

        if not (0.0 <= resolved_score <= 1.0):
            raise ValueError(f"Quiz score must be between 0.0 and 1.0, got {resolved_score}")

        now = datetime.now(timezone.utc).isoformat()
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO quiz_results (id, topic_id, question, student_answer, correct_answer, is_correct, score, feedback, created_at, subject_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    question = excluded.question,
                    student_answer = excluded.student_answer,
                    correct_answer = excluded.correct_answer,
                    is_correct = excluded.is_correct,
                    score = excluded.score,
                    feedback = excluded.feedback,
                    subject_id = excluded.subject_id
                """,
                (
                    resolved_id,
                    resolved_topic_id if resolved_topic_id else None,
                    resolved_q,
                    resolved_student_ans,
                    resolved_correct_ans,
                    1 if resolved_is_correct else 0,
                    resolved_score,
                    resolved_feedback,
                    now,
                    resolved_subject_id
                )
            )

            # Recalculate topic mastery level via Exponential Moving Average (EMA)
            clamped_mastery = resolved_score
            if resolved_topic_id:
                cur = await conn.execute(
                    "SELECT mastery_level FROM topics WHERE id = ?",
                    (resolved_topic_id,)
                )
                topic_row = await cur.fetchone()
                if topic_row:
                    old_mastery = float(topic_row["mastery_level"] if topic_row["mastery_level"] is not None else 0.0)

                    # Check prior quiz count
                    cnt_cur = await conn.execute(
                        "SELECT COUNT(*) as cnt FROM quiz_results WHERE topic_id = ?",
                        (resolved_topic_id,)
                    )
                    cnt_row = await cnt_cur.fetchone()
                    total_quizzes = cnt_row["cnt"] if cnt_row else 1

                    if total_quizzes <= 1 and old_mastery == 0.0:
                        new_mastery = resolved_score
                    else:
                        # EMA: 70% prior mastery, 30% current quiz score
                        new_mastery = (old_mastery * 0.7) + (resolved_score * 0.3)

                    clamped_mastery = max(0.0, min(1.0, round(new_mastery, 4)))

                    has_last_rev = await self._has_topics_last_reviewed_column(conn)
                    if not has_last_rev:
                        try:
                            await conn.execute("ALTER TABLE topics ADD COLUMN last_reviewed TEXT;")
                            has_last_rev = True
                        except Exception:
                            has_last_rev = False

                    if has_last_rev:
                        await conn.execute(
                            "UPDATE topics SET mastery_level = ?, last_reviewed = ?, updated_at = ? WHERE id = ?",
                            (clamped_mastery, now, now, resolved_topic_id)
                        )
                    else:
                        await conn.execute(
                            "UPDATE topics SET mastery_level = ?, updated_at = ? WHERE id = ?",
                            (clamped_mastery, now, resolved_topic_id)
                        )

            return {
                "id": resolved_id,
                "topic_id": resolved_topic_id,
                "question": resolved_q,
                "is_correct": resolved_is_correct,
                "score": resolved_score,
                "mastery_level": clamped_mastery,
                "feedback": resolved_feedback,
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
        cleaned_id = task_id.strip()
        cleaned_title = title.strip()
        if not cleaned_id:
            raise ValueError("Task ID cannot be empty.")
        if not cleaned_title:
            raise ValueError("Task title cannot be empty.")
        cleaned_priority = priority.strip().lower()
        if cleaned_priority not in ("low", "medium", "high"):
            raise ValueError(f"Invalid task priority '{priority}'.")

        now = datetime.now(timezone.utc).isoformat()
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO academic_tasks (id, title, description, subject_id, deadline, priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    subject_id = excluded.subject_id,
                    deadline = excluded.deadline,
                    priority = excluded.priority,
                    status = excluded.status
                """,
                (cleaned_id, cleaned_title, description.strip(), subject_id.strip() if subject_id else None, deadline, cleaned_priority, now)
            )
            return {
                "id": cleaned_id,
                "title": cleaned_title,
                "description": description.strip(),
                "subject_id": subject_id,
                "deadline": deadline,
                "priority": cleaned_priority,
                "status": "pending",
                "created_at": now
            }

    async def update_task_status(self, task_id: str, status: str) -> bool:
        cleaned_id = task_id.strip()
        cleaned_status = status.strip().lower()
        if not cleaned_id:
            raise ValueError("Task ID cannot be empty.")
        if cleaned_status not in ("pending", "in_progress", "completed", "cancelled"):
            raise ValueError(f"Invalid task status '{status}'.")

        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if cleaned_status == "completed" else None
        async with self.transaction() as conn:
            cursor = await conn.execute(
                "UPDATE academic_tasks SET status = ?, completed_at = ? WHERE id = ?",
                (cleaned_status, completed_at, cleaned_id)
            )
            return cursor.rowcount > 0

    async def delete_task(self, task_id: str) -> bool:
        """Delete an academic task by ID."""
        cleaned_id = task_id.strip() if task_id else ""
        if not cleaned_id:
            raise ValueError("Task ID cannot be empty.")
        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM academic_tasks WHERE id = ?", (cleaned_id,))
            return cursor.rowcount > 0

    async def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.get_connection() as conn:
            if status:
                cursor = await conn.execute("SELECT * FROM academic_tasks WHERE status = ? ORDER BY deadline ASC", (status.strip().lower(),))
            else:
                cursor = await conn.execute("SELECT * FROM academic_tasks ORDER BY deadline ASC")
            return [dict(r) for r in await cursor.fetchall()]

    async def get_upcoming_tasks(
        self,
        subject_id: Optional[str] = None,
        limit: int = 5,
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """Fetch pending academic tasks due within the next `days_ahead` days, sorted by closest deadline."""
        from datetime import timedelta
        safe_limit = max(1, limit)
        now_dt = datetime.now(timezone.utc)
        now_str = now_dt.isoformat()
        future_str = (now_dt + timedelta(days=days_ahead)).isoformat()

        async with self.get_connection() as conn:
            query = """
                SELECT * FROM academic_tasks
                WHERE status != 'completed' AND status != 'cancelled'
                  AND deadline IS NOT NULL
                  AND deadline >= ?
                  AND deadline <= ?
            """
            params: List[Any] = [now_str, future_str]
            if subject_id:
                query += " AND (subject_id = ? OR LOWER(subject_id) = LOWER(?))"
                params.extend([subject_id.strip(), subject_id.strip()])

            query += " ORDER BY deadline ASC LIMIT ?"
            params.append(safe_limit)

            cursor = await conn.execute(query, tuple(params))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_overdue_tasks(self) -> List[Dict[str, Any]]:
        """Fetch uncompleted academic tasks whose deadlines have passed."""
        now_str = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM academic_tasks
                WHERE status != 'completed' AND status != 'cancelled'
                  AND deadline IS NOT NULL
                  AND deadline < ?
                ORDER BY deadline ASC
                """,
                (now_str,)
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def add_study_goal(
        self,
        goal_id: str,
        title: str,
        description: str = "",
        target_date: Optional[str] = None,
        status: str = "in_progress",
        progress_percent: float = 0.0,
        subject_id: Optional[str] = None
    ) -> Dict[str, Any]:
        cleaned_id = goal_id.strip()
        cleaned_title = title.strip()
        if not cleaned_id:
            raise ValueError("Goal ID cannot be empty.")
        if not cleaned_title:
            raise ValueError("Goal title cannot be empty.")
        cleaned_status = status.strip().lower()
        if cleaned_status not in ("in_progress", "completed", "archived", "abandoned"):
            raise ValueError(f"Invalid goal status '{status}'.")
        if not (0.0 <= progress_percent <= 100.0):
            raise ValueError(f"Progress percent must be between 0.0 and 100.0, got {progress_percent}")

        now = datetime.now(timezone.utc).isoformat()
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO study_goals (id, title, description, target_date, status, progress_percent, created_at, subject_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    target_date = excluded.target_date,
                    status = excluded.status,
                    progress_percent = excluded.progress_percent,
                    subject_id = excluded.subject_id
                """,
                (cleaned_id, cleaned_title, description.strip(), target_date, cleaned_status, progress_percent, now, subject_id.strip() if subject_id else None)
            )
            return {
                "id": cleaned_id,
                "title": cleaned_title,
                "description": description.strip(),
                "target_date": target_date,
                "status": cleaned_status,
                "progress_percent": progress_percent,
                "created_at": now,
                "subject_id": subject_id
            }

    async def list_study_goals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM study_goals"
        params = ()
        if status:
            query += " WHERE status = ?"
            params = (status.strip().lower(),)
        query += " ORDER BY created_at DESC"
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, params)
            return [dict(r) for r in await cursor.fetchall()]

    async def update_goal_progress(self, goal_id: str, progress_percent: float) -> Dict[str, Any]:
        """Update progress percentage of a study goal, clamping to [0, 100] and auto-completing at 100%."""
        cleaned_id = goal_id.strip() if goal_id else ""
        if not cleaned_id:
            raise ValueError("Goal ID cannot be empty.")

        clamped_percent = max(0.0, min(100.0, round(float(progress_percent), 2)))
        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            cursor = await conn.execute(
                """
                UPDATE study_goals
                SET progress_percent = ?,
                    status = CASE WHEN ? = 100.0 THEN 'completed' ELSE status END
                WHERE id = ?
                """,
                (clamped_percent, clamped_percent, cleaned_id)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Study goal '{cleaned_id}' not found.")

            c2 = await conn.execute("SELECT * FROM study_goals WHERE id = ?", (cleaned_id,))
            row = await c2.fetchone()
            return dict(row) if row else {}

    async def get_active_goals(self, subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve active study goals (status = 'in_progress'), optionally filtered by subject."""
        async with self.get_connection() as conn:
            if subject_id:
                cursor = await conn.execute(
                    "SELECT * FROM study_goals WHERE status = 'in_progress' AND (subject_id = ? OR LOWER(subject_id) = LOWER(?)) ORDER BY created_at DESC",
                    (subject_id.strip(), subject_id.strip())
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM study_goals WHERE status = 'in_progress' ORDER BY created_at DESC"
                )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def delete_study_goal(self, goal_id: str) -> bool:
        """Delete a study goal by ID."""
        cleaned_id = goal_id.strip() if goal_id else ""
        if not cleaned_id:
            raise ValueError("Goal ID cannot be empty.")
        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM study_goals WHERE id = ?", (cleaned_id,))
            return cursor.rowcount > 0

    # --- DOCUMENTS CATALOG ---

    async def upsert_document(
        self,
        doc_id: str,
        filename: str,
        file_path: str,
        file_hash: str,
        file_type: str,
        file_size_bytes: int,
        subject_id: Optional[str] = None,
        title: Optional[str] = None,
        page_count: int = 1,
        chunk_count: int = 0,
        status: str = "indexed",
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        doc_id = doc_id.strip()
        if not doc_id:
            raise ValueError("Document ID cannot be empty.")
        filename = filename.strip()
        if not filename:
            raise ValueError("Filename cannot be empty.")
        file_path = file_path.strip()
        if not file_path:
            raise ValueError("File path cannot be empty.")
        file_hash = file_hash.strip().lower()
        if not file_hash:
            raise ValueError("File hash cannot be empty.")
        clean_type = file_type.lstrip(".").strip().lower()
        if clean_type not in ("pdf", "docx", "txt", "md"):
            raise ValueError(f"Invalid file_type '{file_type}'. Must be one of pdf, docx, txt, md.")
        if file_size_bytes < 0:
            raise ValueError(f"file_size_bytes cannot be negative, got {file_size_bytes}")
        if page_count < 1:
            raise ValueError(f"page_count must be at least 1, got {page_count}")
        if chunk_count < 0:
            raise ValueError(f"chunk_count cannot be negative, got {chunk_count}")
        clean_status = status.strip().lower()
        if clean_status not in ("pending", "processing", "indexed", "failed"):
            raise ValueError(f"Invalid status '{status}'. Must be one of pending, processing, indexed, failed.")

        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            # Resolve subject_id if provided to satisfy foreign key constraints
            valid_subject_id = None
            if subject_id:
                sub_clean = subject_id.strip()
                cur = await conn.execute(
                    "SELECT id FROM subjects WHERE id = ? OR LOWER(name) = LOWER(?) OR LOWER(code) = LOWER(?)",
                    (sub_clean, sub_clean, sub_clean)
                )
                row = await cur.fetchone()
                if row:
                    valid_subject_id = row["id"]
                else:
                    # Auto-create stub subject so foreign key constraint is satisfied
                    slug = re.sub(r'[^a-zA-Z0-9_-]', '_', sub_clean.lower()).strip('_') or "general"
                    try:
                        await conn.execute(
                            "INSERT INTO subjects (id, name, description, created_at, code) VALUES (?, ?, ?, ?, ?)",
                            (slug, sub_clean, f"Subject for {sub_clean}", now, slug.upper()[:20])
                        )
                        valid_subject_id = slug
                    except sqlite3.IntegrityError:
                        cur2 = await conn.execute(
                            "SELECT id FROM subjects WHERE id = ? OR LOWER(name) = LOWER(?) OR LOWER(code) = LOWER(?)",
                            (slug, sub_clean, slug.upper()[:20])
                        )
                        row2 = await cur2.fetchone()
                        valid_subject_id = row2["id"] if row2 else None

            # Check if file_hash already exists under another id
            cursor = await conn.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            existing = await cursor.fetchone()
            target_id = existing["id"] if existing else doc_id

            await conn.execute(
                """
                INSERT INTO documents (
                    id, filename, file_path, file_hash, file_type, file_size_bytes,
                    subject_id, title, page_count, chunk_count, status, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    file_path = excluded.file_path,
                    file_hash = excluded.file_hash,
                    file_type = excluded.file_type,
                    file_size_bytes = excluded.file_size_bytes,
                    subject_id = excluded.subject_id,
                    title = excluded.title,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    target_id, filename, file_path, file_hash, clean_type, file_size_bytes,
                    valid_subject_id, title.strip() if title else None,
                    page_count, chunk_count, clean_status, error_message, now
                )
            )

        res = await self.get_document(target_id)
        assert res is not None
        return res

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        cleaned_id = doc_id.strip()
        if not cleaned_id:
            return None
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM documents WHERE id = ?", (cleaned_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        cleaned_hash = file_hash.strip().lower()
        if not cleaned_hash:
            return None
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM documents WHERE file_hash = ?", (cleaned_hash,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        cleaned_fn = filename.strip()
        if not cleaned_fn:
            return None
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM documents WHERE filename = ? ORDER BY updated_at DESC LIMIT 1", (cleaned_fn,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_documents(
        self,
        subject_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM documents"
        params: List[Any] = []
        clauses: List[str] = []

        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id.strip())
        if status:
            clauses.append("status = ?")
            params.append(status.strip().lower())

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, tuple(params))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def delete_document_record(self, doc_id: str) -> bool:
        cleaned_id = doc_id.strip()
        if not cleaned_id:
            return False
        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM documents WHERE id = ? OR filename = ?", (cleaned_id, cleaned_id))
            return cursor.rowcount > 0

    # --- USER PREFERENCES ---

    async def set_preference(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self.get_connection() as conn:
            await conn.execute(
                "INSERT INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key.strip(), value, now)
            )
            await conn.commit()

    async def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key.strip(),))
            row = await cursor.fetchone()
            return row["value"] if row else default


db = DatabaseManager()
