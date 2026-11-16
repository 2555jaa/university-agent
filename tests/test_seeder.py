"""Tests verifying the academic seeder and CLI database seeding utility."""

import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

from app.config import PROJECT_ROOT, Settings
from app.memory.database import DatabaseManager
from app.memory.seeder import AcademicSeeder


@pytest.fixture
def seeder_fixture(tmp_path):
    """Provides isolated Settings, DatabaseManager, and AcademicSeeder instances."""
    data_dir = tmp_path / "seeder_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    custom_settings = Settings(APP_ENV="testing", DATA_DIR=data_dir, _env_file=None)
    custom_settings.ensure_directories()
    db = DatabaseManager(db_path=custom_settings.SQLITE_PATH, settings_obj=custom_settings)
    seeder = AcademicSeeder(db=db, settings_obj=custom_settings)
    return custom_settings, db, seeder


@pytest.mark.asyncio
async def test_seed_academic_workspace_populates_all_tables(seeder_fixture):
    """Seeding populates all expected academic curriculum tables with valid foreign keys."""
    settings_obj, db, seeder = seeder_fixture

    res = await seeder.seed_academic_workspace(include_sample_activity=True)
    assert res["status"] == "success"
    counts = res["counts"]

    assert counts["subjects"] == 4
    assert counts["topics"] == 12
    assert counts["documents"] == 4
    assert counts["sessions"] == 2
    assert counts["messages"] == 4
    assert counts["quizzes"] == 3
    assert counts["tasks"] == 4
    assert counts["goals"] == 3

    assert res["created"]["subjects"] == 4
    assert res["existing"]["subjects"] == 0
    assert res["total"]["subjects"] == 4

    # Verify foreign key linkages and data integrity
    async with db.get_connection() as conn:
        # Subjects exist
        sub_rows = await (await conn.execute("SELECT id, code, name FROM subjects ORDER BY code")).fetchall()
        codes = [r["code"] for r in sub_rows]
        assert codes == ["AI301", "CS201", "MT204", "PH102"]

        # Topics point to valid subjects
        top_rows = await (await conn.execute("SELECT t.id, t.name, s.code FROM topics t JOIN subjects s ON t.subject_id = s.id")).fetchall()
        assert len(top_rows) == 12

        # Documents catalog has valid entries
        doc_rows = await (await conn.execute("SELECT id, filename, subject_id, status FROM documents")).fetchall()
        assert len(doc_rows) == 4
        for d in doc_rows:
            assert d["status"] == "indexed"
            assert d["subject_id"] in codes

        # Messages point to sessions
        msg_rows = await (await conn.execute("SELECT m.id, m.role, m.content, s.title FROM messages m JOIN learning_sessions s ON m.session_id = s.id")).fetchall()
        assert len(msg_rows) == 4


@pytest.mark.asyncio
async def test_seeder_strict_idempotency_consecutive_runs(seeder_fixture):
    """Running seed_academic_workspace 3 times consecutively produces identical table counts."""
    settings_obj, db, seeder = seeder_fixture

    tables = [
        "subjects", "topics", "documents", "learning_sessions",
        "messages", "quiz_results", "academic_tasks", "study_goals"
    ]

    async def get_all_counts():
        counts = {}
        async with db.get_connection() as conn:
            for t in tables:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {t}")
                counts[t] = (await cursor.fetchone())[0]
        return counts

    # Run 1
    res1 = await seeder.seed_academic_workspace(include_sample_activity=True)
    counts_run_1 = await get_all_counts()
    assert res1["created"]["subjects"] == 4
    assert res1["existing"]["subjects"] == 0

    # Run 2
    res2 = await seeder.seed_academic_workspace(include_sample_activity=True)
    counts_run_2 = await get_all_counts()
    assert res2["created"]["subjects"] == 0
    assert res2["existing"]["subjects"] == 4

    # Run 3
    res3 = await seeder.seed_academic_workspace(include_sample_activity=True)
    counts_run_3 = await get_all_counts()
    assert res3["created"]["subjects"] == 0
    assert res3["existing"]["subjects"] == 4

    assert counts_run_1 == counts_run_2, f"Run 2 mutated row counts: {counts_run_1} vs {counts_run_2}"
    assert counts_run_2 == counts_run_3, f"Run 3 mutated row counts: {counts_run_2} vs {counts_run_3}"


@pytest.mark.asyncio
async def test_seeder_computes_sha256_when_physical_files_exist(seeder_fixture):
    """If physical files exist in DOCUMENTS_DIR, seeder computes their actual SHA-256 and size."""
    settings_obj, db, seeder = seeder_fixture

    # Write a dummy physical lecture note
    test_pdf_content = b"%PDF-1.4 simulated test content for virtual memory notes"
    expected_hash = hashlib.sha256(test_pdf_content).hexdigest()
    expected_size = len(test_pdf_content)

    test_file = settings_obj.DOCUMENTS_DIR / "os_virtual_memory.pdf"
    test_file.write_bytes(test_pdf_content)

    await seeder.seed_academic_workspace(include_sample_activity=False)

    doc_record = await db.get_document_by_filename("os_virtual_memory.pdf")
    assert doc_record is not None
    assert doc_record["file_hash"] == expected_hash
    assert doc_record["file_size_bytes"] == expected_size
    assert doc_record["subject_id"] == "CS201"


def test_cli_script_seed_database_execution(tmp_path):
    """Executing scripts/seed_database.py via CLI populates the target database cleanly."""
    tmp_db = tmp_path / "cli_seed_test.db"
    script_path = PROJECT_ROOT / "scripts" / "seed_database.py"

    # 1. Run seeder CLI
    cmd = [
        sys.executable,
        str(script_path),
        "--env", "testing",
        "--db-path", str(tmp_db)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"CLI seed failed: {res.stderr}\nOutput: {res.stdout}"
    assert "[SUCCESS] Seeding complete" in res.stdout

    # Verify rows exist in created database
    mgr = DatabaseManager(db_path=tmp_db)
    with mgr.get_connection_sync() as conn:
        sub_count = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        assert sub_count == 4

    # 2. Run verify-only flag
    cmd_verify = [
        sys.executable,
        str(script_path),
        "--env", "testing",
        "--db-path", str(tmp_db),
        "--verify-only"
    ]
    res_verify = subprocess.run(cmd_verify, capture_output=True, text=True, timeout=30)
    assert res_verify.returncode == 0
    assert "DATABASE STATUS" in res_verify.stdout
    assert "subjects          : 4" in res_verify.stdout
