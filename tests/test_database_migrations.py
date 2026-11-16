"""Comprehensive tests for SQLite migrations, pragmas, constraints, and indexes."""

import shutil
import sqlite3
import pytest
import aiosqlite

from app.memory.database import DatabaseManager, _migration_001_initial_schema


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "migration_test.db"


@pytest.fixture
def db_manager(temp_db_path):
    mgr = DatabaseManager(db_path=temp_db_path)
    mgr.init_db_sync()
    return mgr


def test_pragmas_configured_on_sync_connection(db_manager):
    """Verify foreign_keys, journal_mode, synchronous, and busy_timeout are active synchronously."""
    with db_manager.get_connection_sync() as conn:
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]

        assert fk == 1, "Foreign keys must be ON"
        assert journal.lower() == "wal", "Journal mode must be WAL"
        assert sync == 1, "Synchronous must be NORMAL (1)"
        assert timeout == 5000, "Busy timeout must be 5000ms"


@pytest.mark.asyncio
async def test_pragmas_configured_on_async_connection(db_manager):
    """Verify foreign_keys, journal_mode, synchronous, and busy_timeout are active asynchronously."""
    async with db_manager.get_connection() as conn:
        fk_row = await (await conn.execute("PRAGMA foreign_keys;")).fetchone()
        journal_row = await (await conn.execute("PRAGMA journal_mode;")).fetchone()
        sync_row = await (await conn.execute("PRAGMA synchronous;")).fetchone()
        timeout_row = await (await conn.execute("PRAGMA busy_timeout;")).fetchone()

        assert fk_row[0] == 1, "Async foreign keys must be ON"
        assert journal_row[0].lower() == "wal", "Async journal mode must be WAL"
        assert sync_row[0] == 1, "Async synchronous must be NORMAL (1)"
        assert timeout_row[0] == 5000, "Async busy timeout must be 5000ms"


def test_migration_engine_idempotency_and_version(db_manager):
    """Running migrations multiple times is idempotent and preserves the current schema version."""
    v1 = db_manager.get_schema_version_sync()
    assert v1 == 3, f"Expected schema version 3, got {v1}"

    # Re-run migrations
    v2 = db_manager.run_migrations_sync()
    assert v2 == 3


@pytest.mark.asyncio
async def test_check_health_reports_schema_version(db_manager):
    """DatabaseManager.check_health() includes schema_version and wal_mode status."""
    health = await db_manager.check_health()
    assert health["status"] == "ok"
    assert health["schema_version"] == 3
    assert health["wal_mode"] is True


def test_legacy_database_upgrade_without_data_loss(tmp_path):
    """A legacy pre-migration database is upgraded to version 2 without losing existing rows."""
    legacy_db_file = tmp_path / "legacy_pre_migration.db"
    
    # 1. Create legacy DB using migration 1 only (no schema_migrations table, no v2 constraints)
    conn = sqlite3.connect(legacy_db_file)
    _migration_001_initial_schema(conn)
    conn.execute("INSERT INTO subjects (id, name, description, created_at) VALUES ('cs101', 'Intro to CS', 'Desc', '2026-01-01');")
    conn.execute("INSERT INTO topics (id, subject_id, name, mastery_level, notes, updated_at) VALUES ('top1', 'cs101', 'Recursion', 0.8, 'Notes', '2026-01-01');")
    conn.execute("INSERT INTO learning_sessions (id, session_type, title, started_at) VALUES ('sess1', 'tutor', 'Study Session', '2026-01-01');")
    conn.execute("INSERT INTO messages (id, session_id, role, content, mode, timestamp) VALUES ('msg1', 'sess1', 'user', 'Hello tutor', 'tutor', '2026-01-01');")
    conn.execute("INSERT INTO academic_tasks (id, title, status, priority, created_at) VALUES ('tsk1', 'Homework 1', 'completed', 'high', '2026-01-01');")
    conn.commit()
    conn.close()

    # 2. Run DatabaseManager on this legacy DB
    mgr = DatabaseManager(db_path=legacy_db_file)
    ver = mgr.run_migrations_sync()
    assert ver == 3

    # 3. Verify all existing rows are preserved
    with mgr.get_connection_sync() as c:
        assert c.execute("SELECT name FROM subjects WHERE id='cs101'").fetchone()[0] == "Intro to CS"
        assert c.execute("SELECT name FROM topics WHERE id='top1'").fetchone()[0] == "Recursion"
        assert c.execute("SELECT content FROM messages WHERE id='msg1'").fetchone()[0] == "Hello tutor"
        assert c.execute("SELECT title FROM academic_tasks WHERE id='tsk1'").fetchone()[0] == "Homework 1"


def test_foreign_key_constraints_enforced(db_manager):
    """Inserting into topics with a non-existent subject_id raises IntegrityError."""
    with pytest.raises(sqlite3.IntegrityError):
        with db_manager.get_connection_sync() as conn:
            conn.execute(
                "INSERT INTO topics (id, subject_id, name, mastery_level, notes, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("top_invalid", "non_existent_subject", "Algorithms", 0.5, "", "2026-01-01")
            )


@pytest.mark.asyncio
async def test_unique_subject_and_topic_constraints(db_manager):
    """Case-insensitive duplicate subject names and duplicate topics under same subject are rejected."""
    await db_manager.add_subject("phys1", "Physics I", "Mechanics")

    # Duplicate subject name (case-insensitive)
    with pytest.raises(sqlite3.IntegrityError):
        with db_manager.get_connection_sync() as conn:
            conn.execute(
                "INSERT INTO subjects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                ("phys2", "physics i", "Another", "2026-01-01")
            )

    # Add topic under phys1
    await db_manager.add_or_update_topic("top_kin", "phys1", "Kinematics", 0.5)

    # Duplicate topic name under the same subject (case-insensitive)
    with pytest.raises(sqlite3.IntegrityError):
        with db_manager.get_connection_sync() as conn:
            conn.execute(
                "INSERT INTO topics (id, subject_id, name, mastery_level, notes, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("top_kin2", "phys1", "kinematics", 0.5, "", "2026-01-01")
            )


def test_check_constraints_enforce_valid_domains(db_manager):
    """Invalid roles, statuses, priorities, and out-of-range values are rejected by CHECK constraints."""
    with db_manager.get_connection_sync() as conn:
        # Create session first
        conn.execute("INSERT INTO learning_sessions (id, session_type, title, started_at) VALUES ('s_test', 'tutor', 'Test', '2026-01-01');")

        # 1. Invalid message role
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, mode, timestamp) VALUES ('m_bad', 's_test', 'attacker_role', 'hi', 'tutor', '2026-01-01');"
            )

        # 2. Invalid task status
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO academic_tasks (id, title, status, priority, created_at) VALUES ('t_bad', 'Task', 'invalid_status', 'low', '2026-01-01');"
            )

        # 3. Invalid task priority
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO academic_tasks (id, title, status, priority, created_at) VALUES ('t_bad2', 'Task', 'pending', 'super_urgent', '2026-01-01');"
            )

        # 4. Out-of-range mastery_level (> 1.0)
        conn.execute("INSERT INTO subjects (id, name, created_at) VALUES ('math1', 'Calculus', '2026-01-01');")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO topics (id, subject_id, name, mastery_level, updated_at) VALUES ('top_bad', 'math1', 'Limits', 1.5, '2026-01-01');"
            )


def test_performance_indexes_exist(db_manager):
    """All performance indexes exist in sqlite_master."""
    with db_manager.get_connection_sync() as conn:
        indexes = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }

        expected = {
            "idx_subjects_id_lower",
            "idx_subjects_name_lower",
            "idx_subjects_code_lower",
            "idx_topics_subject_name_lower",
            "idx_topics_subject",
            "idx_sessions_subject_updated",
            "idx_sessions_started_at",
            "idx_messages_session_time",
            "idx_quizzes_subject_topic",
            "idx_quizzes_topic_created",
            "idx_tasks_status_deadline",
            "idx_tasks_subject",
            "idx_study_goals_status",
            "idx_documents_hash",
            "idx_documents_subject",
            "idx_documents_status_updated",
            "idx_documents_filename",
        }

        missing = expected - indexes
        assert not missing, f"Missing expected indexes: {missing}"


@pytest.mark.asyncio
async def test_documents_catalog_crud_lifecycle(db_manager):
    """Full CRUD lifecycle on documents metadata catalog table."""
    # 1. Subject link
    await db_manager.add_subject("cs201", "Algorithms", "Advanced CS")

    # 2. Insert document
    doc = await db_manager.upsert_document(
        doc_id="doc_algo_01",
        filename="sorting_algorithms.pdf",
        file_path="/data/documents/sorting_algorithms.pdf",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        file_type="pdf",
        file_size_bytes=1048576,
        subject_id="cs201",
        title="Sorting Algorithms Lecture",
        page_count=25,
        chunk_count=120,
        status="pending"
    )
    assert doc["id"] == "doc_algo_01"
    assert doc["status"] == "pending"
    assert doc["subject_id"] == "cs201"
    assert doc["page_count"] == 25

    # 3. Lookup by hash and filename
    by_hash = await db_manager.get_document_by_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert by_hash is not None
    assert by_hash["id"] == "doc_algo_01"

    by_fn = await db_manager.get_document_by_filename("sorting_algorithms.pdf")
    assert by_fn is not None
    assert by_fn["id"] == "doc_algo_01"

    # 4. Status transition pending -> processing -> indexed
    updated_doc = await db_manager.upsert_document(
        doc_id="doc_algo_01",
        filename="sorting_algorithms.pdf",
        file_path="/data/documents/sorting_algorithms.pdf",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        file_type="pdf",
        file_size_bytes=1048576,
        subject_id="cs201",
        title="Sorting Algorithms Lecture",
        page_count=25,
        chunk_count=120,
        status="indexed"
    )
    assert updated_doc["status"] == "indexed"

    # 5. List with filters
    docs_by_sub = await db_manager.list_documents(subject_id="cs201")
    assert len(docs_by_sub) == 1
    assert docs_by_sub[0]["id"] == "doc_algo_01"

    docs_failed = await db_manager.list_documents(status="failed")
    assert len(docs_failed) == 0

    # 6. Delete document record
    deleted = await db_manager.delete_document_record("doc_algo_01")
    assert deleted is True
    assert await db_manager.get_document("doc_algo_01") is None


@pytest.mark.asyncio
async def test_atomic_transaction_rollback(db_manager):
    """Simulated failure inside transaction rolls back all intermediate operations."""
    await db_manager.add_subject("bio101", "Biology 101")
    await db_manager.add_or_update_topic("top_genetics", "bio101", "Genetics", mastery=0.2)

    # Attempt a transaction that inserts a quiz result then raises an error
    with pytest.raises(RuntimeError):
        async with db_manager.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO quiz_results (id, topic_id, question, student_answer, correct_answer, is_correct, score, feedback, created_at)
                VALUES ('q_atomic_fail', 'top_genetics', 'Question', 'A', 'A', 1, 1.0, 'Great', '2026-01-01')
                """
            )
            raise RuntimeError("Simulated crash mid-transaction!")

    # Verify quiz result was rolled back completely
    async with db_manager.get_connection() as conn:
        row = await (await conn.execute("SELECT * FROM quiz_results WHERE id='q_atomic_fail'")).fetchone()
        assert row is None, "Transaction rollback failed: row should not exist!"

        # Topic mastery should remain unchanged at 0.2
        topic_row = await (await conn.execute("SELECT mastery_level FROM topics WHERE id='top_genetics'")).fetchone()
        assert topic_row[0] == 0.2


@pytest.mark.asyncio
async def test_input_validation_and_whitespace_rejection(db_manager):
    """Empty strings and invalid domain values are strictly rejected before database execution."""
    # Blank subject name
    with pytest.raises(ValueError) as exc:
        await db_manager.add_subject("s_blank", "   ")
    assert "empty" in str(exc.value).lower()

    # Out of range mastery level
    with pytest.raises(ValueError):
        await db_manager.add_or_update_topic("t1", "s1", "Topic", mastery=1.5)

    with pytest.raises(ValueError):
        await db_manager.add_or_update_topic("t1", "s1", "Topic", mastery=-0.1)

    # Invalid document file_type
    with pytest.raises(ValueError) as exc:
        await db_manager.upsert_document(
            doc_id="d1", filename="malware.exe", file_path="/tmp/malware.exe",
            file_hash="123456", file_type="exe", file_size_bytes=100
        )
    assert "file_type" in str(exc.value)

    # Negative file size
    with pytest.raises(ValueError):
        await db_manager.upsert_document(
            doc_id="d1", filename="notes.txt", file_path="/tmp/notes.txt",
            file_hash="123456", file_type="txt", file_size_bytes=-10
        )

    # Invalid message role
    with pytest.raises(ValueError):
        await db_manager.save_message("m1", "sess1", "attacker", "hello")


@pytest.mark.asyncio
async def test_idempotent_subject_and_topic_normalization(db_manager):
    """Case and whitespace variations resolve idempotently without duplicating records."""
    # First creation
    sub1 = await db_manager.add_subject("chem101", "Chemistry I", "Basic Chemistry")
    assert sub1["name"] == "Chemistry I"
    assert sub1["code"] == "CHEM101"

    # Repeated creation with whitespace and case variations resolves to existing record
    sub2 = await db_manager.add_subject("CHEM101", " chemistry i ", "Different desc")
    assert sub2["id"] == "chem101"

    subjects = await db_manager.list_subjects()
    chem_subjects = [s for s in subjects if "chemistry" in s["name"].lower()]
    assert len(chem_subjects) == 1, "Should not create duplicate subject for different casing/whitespace"

