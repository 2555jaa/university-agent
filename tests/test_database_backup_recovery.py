"""Comprehensive tests for database backup, restore, corruption recovery, and performance verification."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from app.memory.database import DatabaseManager


@pytest.fixture
def temp_db_dir(tmp_path):
    db_dir = tmp_path / "agent_data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


@pytest.fixture
def db_manager(temp_db_dir):
    mgr = DatabaseManager(db_path=temp_db_dir / "university_agent.db")
    mgr.init_db_sync()
    return mgr


# --- 1. INTEGRITY CHECK TESTS ---


def test_integrity_check_healthy_database(db_manager):
    """Verify integrity report on a clean, initialized database."""
    report = db_manager.verify_integrity_sync()
    assert report["ok"] is True
    assert report["integrity_check"] == "ok"
    assert report["foreign_key_violations"] == []
    assert report["schema_version"] == 3


@pytest.mark.asyncio
async def test_integrity_check_async(db_manager):
    """Verify asynchronous integrity checking matches synchronous report."""
    report = await db_manager.verify_integrity()
    assert report["ok"] is True
    assert report["integrity_check"] == "ok"
    assert report["foreign_key_violations"] == []
    assert report["schema_version"] == 3

    health = await db_manager.check_health()
    assert health["status"] == "ok"
    assert health["integrity"] == "ok"
    assert health["schema_version"] == 3


def test_integrity_check_detects_foreign_key_violations(db_manager):
    """Verify that foreign key violations cause verify_integrity_sync to report failure."""
    # Temporarily bypass FK pragma on a direct connection to insert orphaned topic
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute(
            "INSERT INTO topics (id, subject_id, name, mastery_level, updated_at) VALUES ('t-orphan', 'sub-missing', 'Orphan Topic', 0.5, '2026-11-16T00:00:00Z');"
        )
        conn.commit()

    report = db_manager.verify_integrity_sync()
    assert report["ok"] is False
    assert len(report["foreign_key_violations"]) > 0
    assert "Foreign key check failed" in report.get("error", "")


def test_integrity_check_handles_corrupt_database_cleanly(temp_db_dir):
    """Verify that corrupt databases or missing files return structured errors without crashing."""
    corrupt_db_path = temp_db_dir / "corrupted.db"
    corrupt_db_path.write_bytes(b"THIS_IS_NOT_A_VALID_SQLITE_DATABASE_HEADER_AT_ALL_000000000")

    mgr = DatabaseManager(db_path=corrupt_db_path)
    report = mgr.verify_integrity_sync()
    assert report["ok"] is False
    assert "error" in report
    assert report["schema_version"] == 0

    # Non-existent database path
    missing_mgr = DatabaseManager(db_path=temp_db_dir / "does_not_exist.db")
    missing_report = missing_mgr.verify_integrity_sync()
    assert missing_report["ok"] is False
    assert missing_report["integrity_check"] == "file_not_found"


# --- 2. HOT-SAFE BACKUP TESTS ---


@pytest.mark.asyncio
async def test_hot_backup_wal_mode_and_manifest(db_manager, temp_db_dir):
    """Verify hot online backup creates a verified file with matching manifest metadata and row counts."""
    # Populate data across tables
    await db_manager.add_subject("CS101", "Computer Systems", "Introduction to systems")
    await db_manager.add_or_update_topic("T1", "CS101", "Memory Hierarchy", notes="Cache and RAM")
    await db_manager.create_session("s-101", "tutor", "Reviewing Caching")
    await db_manager.save_message("m-1", "s-101", "user", "What is L1 cache?")
    await db_manager.save_message("m-2", "s-101", "assistant", "L1 is the smallest and fastest cache.")
    await db_manager.record_quiz_result("q-1", "T1", "What is L1?", "Cache", "Cache", True, 1.0, "Correct", "CS101")
    await db_manager.add_task("task-1", "Read Chapter 4", priority="high")
    await db_manager.add_study_goal("goal-1", "Master Cache Architecture", progress_percent=50.0)
    await db_manager.upsert_document(
        doc_id="doc-1",
        filename="systems.pdf",
        file_path="/data/documents/systems.pdf",
        file_hash="abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
        file_type="pdf",
        file_size_bytes=10240,
        page_count=10,
        chunk_count=20
    )

    backup_dir = temp_db_dir / "custom_backups"
    manifest = db_manager.create_backup_sync(backup_dir=backup_dir, label="unit_test")

    # Verify manifest fields
    assert manifest["backup_file"].startswith("university_agent_unit_test_")
    assert manifest["label"] == "unit_test"
    assert manifest["schema_version"] == 3
    assert manifest["integrity"] == "ok"
    assert manifest["size_bytes"] > 0

    backup_path = Path(manifest["backup_path"])
    meta_path = Path(manifest["meta_path"])
    assert backup_path.exists()
    assert meta_path.exists()

    # Verify SHA-256 digest
    digest = hashlib.sha256(backup_path.read_bytes()).hexdigest()
    assert digest == manifest["sha256"]

    # Verify companion metadata matches
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_file_data = json.load(f)
    assert meta_file_data == manifest

    # Verify table row counts
    counts = manifest["table_counts"]
    assert counts.get("subjects") == 1
    assert counts.get("topics") == 1
    assert counts.get("learning_sessions") == 1
    assert counts.get("messages") == 2
    assert counts.get("quiz_results") == 1
    assert counts.get("academic_tasks") == 1
    assert counts.get("study_goals") == 1
    assert counts.get("documents") == 1

    # Verify backup file itself passes integrity check standalone
    backup_integrity = db_manager.verify_integrity_sync(backup_path)
    assert backup_integrity["ok"] is True
    assert backup_integrity["schema_version"] == 3

    # Test async create_backup
    async_manifest = await db_manager.create_backup(backup_dir=backup_dir, label="async_test")
    assert Path(async_manifest["backup_path"]).exists()
    assert async_manifest["integrity"] == "ok"


# --- 3. ATOMIC RESTORE TESTS ---


@pytest.mark.asyncio
async def test_atomic_restore_preserves_data_and_cleans_sidecars(db_manager, temp_db_dir):
    """Verify restoring from backup replaces active database state atomically and clears stale WAL sidecars."""
    # State A: Initial baseline
    await db_manager.add_subject("CS101", "Computer Systems")
    await db_manager.create_session("sess-orig", "tutor", "Original Session")
    await db_manager.save_message("msg-orig", "sess-orig", "user", "Original content")

    backup_manifest = db_manager.create_backup_sync(label="baseline")
    backup_path = Path(backup_manifest["backup_path"])

    # State B: Mutate active DB (add new data, delete original session)
    await db_manager.add_subject("CS202", "Algorithms & Complexity")
    async with db_manager.transaction() as conn:
        await conn.execute("DELETE FROM messages WHERE id = 'msg-orig';")
        await conn.execute("DELETE FROM learning_sessions WHERE id = 'sess-orig';")

    # Confirm mutations in active DB
    msgs_b = await db_manager.get_session_history("sess-orig")
    assert len(msgs_b) == 0
    sub_b = await db_manager.list_subjects()
    assert len(sub_b) == 2

    # Ensure WAL sidecar is present
    wal_file = db_manager.db_path.with_name(f"{db_manager.db_path.name}-wal")
    if not wal_file.exists():
        with db_manager.get_connection_sync() as conn:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")

    # Perform atomic restore
    restore_report = db_manager.restore_backup_sync(backup_path)
    assert restore_report["ok"] is True
    assert restore_report["schema_version"] == 3
    assert restore_report["integrity"]["ok"] is True
    assert restore_report["pre_restore_snapshot"] is not None
    assert Path(restore_report["pre_restore_snapshot"]).exists()

    # Verify active DB restored to State A
    sub_restored = await db_manager.list_subjects()
    assert len(sub_restored) == 1
    assert sub_restored[0]["id"] == "CS101"

    msgs_restored = await db_manager.get_session_history("sess-orig")
    assert len(msgs_restored) == 1
    assert msgs_restored[0]["content"] == "Original content"

    # Test async restore_backup method
    async_restore_report = await db_manager.restore_backup(backup_path)
    assert async_restore_report["ok"] is True


@pytest.mark.asyncio
async def test_restore_rejects_corrupted_backup_leaving_active_db_intact(db_manager, temp_db_dir):
    """Verify that corrupt or tampered backup files are rejected with ValueError and active DB remains untouched."""
    await db_manager.add_subject("CS303", "Operating Systems")
    await db_manager.create_session("sess-protect", "tutor", "Protected Session")

    # Create good backup then tamper with it
    manifest = db_manager.create_backup_sync(label="tamper_test")
    backup_path = Path(manifest["backup_path"])
    meta_path = Path(manifest["meta_path"])

    # 1. Tamper backup file content (SHA-256 mismatch against manifest)
    with open(backup_path, "r+b") as f:
        f.seek(0)
        f.write(b"CORRUPTED_HEADER_INVALID_SQLITE_MAGIC")

    with pytest.raises(ValueError, match="Backup SHA-256 mismatch"):
        db_manager.restore_backup_sync(backup_path)

    # 2. Tamper manifest to match bad SHA-256, but backup file fails SQLite integrity check
    tampered_digest = hashlib.sha256(backup_path.read_bytes()).hexdigest()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    meta_data["sha256"] = tampered_digest
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    with pytest.raises(ValueError, match="Backup file failed integrity verification"):
        db_manager.restore_backup_sync(backup_path)

    # 3. Corrupt backup without any metadata manifest
    corrupt_orphan = temp_db_dir / "corrupt_orphan.db"
    corrupt_orphan.write_bytes(b"NOT_A_SQLITE_FILE_AT_ALL")
    with pytest.raises(ValueError, match="Backup file failed integrity verification"):
        db_manager.restore_backup_sync(corrupt_orphan)

    # Non-existent backup file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        db_manager.restore_backup_sync(temp_db_dir / "ghost_backup.db")

    # Verify active DB is 100% UNTOUCHED
    subjects = await db_manager.list_subjects()
    assert len(subjects) == 1
    assert subjects[0]["id"] == "CS303"


# --- 4. CORRUPTION RECOVERY TESTS ---


@pytest.mark.asyncio
async def test_recover_or_restore_handles_checkpoint_and_corrupt_database(db_manager, temp_db_dir):
    """Verify recovery via checkpoint for healthy DB, and automatic fallback restore when corrupted."""
    # Populate data and create valid backup
    await db_manager.add_subject("CS404", "Computer Networks")
    await db_manager.create_session("sess-net", "tutor", "Networking Session")

    manifest = db_manager.create_backup_sync(label="recovery_base")
    good_backup_path = Path(manifest["backup_path"])

    # Strategy 1: Healthy database uses checkpoint
    rec_checkpoint = db_manager.recover_or_restore_sync(fallback_backup_path=good_backup_path)
    assert rec_checkpoint["recovered"] is True
    assert rec_checkpoint["strategy"] == "checkpoint"
    assert rec_checkpoint["integrity"]["ok"] is True

    # Strategy 2: Corrupted database auto-recovers from fallback backup
    # Write garbage into active DB file
    with open(db_manager.db_path, "wb") as f:
        f.write(b"CORRUPTED_DISK_IMAGE_INVALID_HEADER")

    assert db_manager.verify_integrity_sync()["ok"] is False

    # Execute recovery with fallback
    rec_restore = db_manager.recover_or_restore_sync(fallback_backup_path=good_backup_path)
    assert rec_restore["recovered"] is True
    assert rec_restore["strategy"] == "restored_from_backup"
    assert rec_restore["details"]["ok"] is True

    # Verify active database is healthy and has restored data
    assert db_manager.verify_integrity_sync()["ok"] is True
    subjects = await db_manager.list_subjects()
    assert len(subjects) == 1
    assert subjects[0]["id"] == "CS404"

    # Strategy 3: Corrupted database with no fallback backup
    with open(db_manager.db_path, "wb") as f:
        f.write(b"CORRUPTED_AGAIN_NO_BACKUP")

    rec_fail = db_manager.recover_or_restore_sync(fallback_backup_path=None)
    assert rec_fail["recovered"] is False
    assert rec_fail["strategy"] == "corrupted_no_backup"

    # Test async wrapper
    async_rec = await db_manager.recover_or_restore(fallback_backup_path=good_backup_path)
    assert async_rec["recovered"] is True


# --- 5. PERFORMANCE BENCHMARKS & QUERY PLAN TESTS ---


@pytest.mark.asyncio
async def test_database_performance_and_query_plans(db_manager):
    """Benchmark bulk inserts and verify that critical queries use appropriate indexes."""
    await db_manager.add_subject("CS500", "Advanced Distributed Systems")
    await db_manager.add_or_update_topic("T500", "CS500", "Consensus Algorithms")
    await db_manager.create_session("sess-bulk", "tutor", "Bulk Message Test")

    # Benchmark: Insert 500 messages inside an atomic transaction
    start_time = time.perf_counter()
    async with db_manager.transaction() as conn:
        for i in range(500):
            ts = f"2026-11-16T10:{i // 60:02d}:{i % 60:02d}Z"
            await conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (f"m-bulk-{i}", "sess-bulk", "user" if i % 2 == 0 else "assistant", f"Message payload {i}", ts)
            )
    elapsed_bulk_insert = time.perf_counter() - start_time
    assert elapsed_bulk_insert < 2.0, f"Bulk inserting 500 messages took {elapsed_bulk_insert:.3f}s (budget: 2.0s)"

    # Benchmark: Fetch all 500 messages
    fetch_start = time.perf_counter()
    messages = await db_manager.get_session_history("sess-bulk", limit=500)
    elapsed_fetch = time.perf_counter() - fetch_start
    assert len(messages) == 500
    assert elapsed_fetch < 0.1, f"Fetching 500 messages took {elapsed_fetch:.3f}s (budget: 0.1s)"

    # Populate additional tables for index verification
    for i in range(10):
        await db_manager.add_task(f"t-bulk-{i}", f"Task {i}", priority="medium")
        await db_manager.upsert_document(
            doc_id=f"doc-bulk-{i}",
            filename=f"doc_{i}.txt",
            file_path=f"/data/documents/doc_{i}.txt",
            file_hash=f"hash{i:064d}",
            file_type="txt",
            file_size_bytes=100
        )
        await db_manager.record_quiz_result(
            f"q-bulk-{i}", "T500", f"Question {i}", "Ans", "Ans", True, 1.0, "Good", "CS500"
        )

    # Verify query plans using EXPLAIN QUERY PLAN
    with db_manager.get_connection_sync() as conn:
        # 1. messages indexed by (session_id, timestamp)
        plan_msgs = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            ("sess-bulk",)
        ).fetchall()
        msgs_details = " ".join([row["detail"] for row in plan_msgs])
        assert "USING INDEX idx_messages_session_time" in msgs_details, f"Messages query didn't use index: {msgs_details}"

        # 2. academic_tasks indexed by (status, deadline)
        plan_tasks = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM academic_tasks WHERE status = ? ORDER BY deadline ASC",
            ("pending",)
        ).fetchall()
        tasks_details = " ".join([row["detail"] for row in plan_tasks])
        assert "USING INDEX idx_tasks_status_deadline" in tasks_details, f"Tasks query didn't use index: {tasks_details}"

        # 3. documents unique index on file_hash
        plan_docs = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM documents WHERE file_hash = ?",
            ("hash0000000000000000000000000000000000000000000000000000000000000000",)
        ).fetchall()
        docs_details = " ".join([row["detail"] for row in plan_docs])
        assert "USING INDEX idx_documents_hash" in docs_details, f"Documents hash query didn't use index: {docs_details}"

        # 4. quiz_results indexed by (topic_id, created_at)
        plan_quizzes = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM quiz_results WHERE topic_id = ? ORDER BY created_at ASC",
            ("T500",)
        ).fetchall()
        quizzes_details = " ".join([row["detail"] for row in plan_quizzes])
        assert "USING INDEX idx_quizzes_topic_created" in quizzes_details, f"Quizzes query didn't use index: {quizzes_details}"
