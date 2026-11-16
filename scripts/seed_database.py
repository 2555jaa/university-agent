#!/usr/bin/env python3
"""CLI utility to seed the University AI Agent database with realistic academic data."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings
from app.memory.database import DatabaseManager
from app.memory.seeder import AcademicSeeder


def parse_args():
    parser = argparse.ArgumentParser(description="Seed University AI Agent database with academic data.")
    parser.add_argument("--env", type=str, default="development", choices=["development", "testing", "production"],
                        help="Environment profile to use for settings (default: development).")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Custom path to SQLite database file.")
    parser.add_argument("--no-activity", action="store_true",
                        help="Seed only subjects, topics, and documents without sample sessions/tasks/quizzes.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Inspect and display table counts without modifying database.")
    return parser.parse_args()


async def inspect_db(db: DatabaseManager) -> None:
    """Print current table row counts."""
    async with db.get_connection() as conn:
        tables = [
            "subjects", "topics", "documents", "learning_sessions",
            "messages", "quiz_results", "academic_tasks", "study_goals"
        ]
        print("\n--- DATABASE STATUS ---")
        print(f"Path: {db.db_path}")
        print(f"Schema Version: {db.get_schema_version_sync()}")
        print("-" * 25)
        for t in tables:
            try:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {t}")
                count = (await cursor.fetchone())[0]
                print(f"  {t:<18}: {count}")
            except Exception as e:
                print(f"  {t:<18}: error ({e})")
        print("-" * 25)


async def main_async(args):
    db_path = Path(args.db_path).resolve() if args.db_path else None
    overrides = {}
    if db_path:
        overrides["SQLITE_PATH"] = db_path
        overrides["DATA_DIR"] = db_path.parent

    settings_obj = get_settings(env_override=args.env, **overrides)
    db = DatabaseManager(db_path=db_path or settings_obj.SQLITE_PATH, settings_obj=settings_obj)

    if args.verify_only:
        await inspect_db(db)
        return

    print(f"Seeding academic workspace in '{args.env}' environment...")
    print(f"Target SQLite database: {db.db_path}")

    seeder = AcademicSeeder(db=db, settings_obj=settings_obj)
    res = await seeder.seed_academic_workspace(include_sample_activity=not args.no_activity)

    print("\n[SUCCESS] Seeding complete:")
    for k, v in res["counts"].items():
        print(f"  {k:<12}: {v}")

    await inspect_db(db)


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except Exception as e:
        print(f"\n[ERROR] Seeding failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
