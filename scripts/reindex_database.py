#!/usr/bin/env python3
"""CLI utility to reindex documents and recover ChromaDB vector store from SQLite catalog."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.memory.database import DatabaseManager
from app.knowledge.vector_store import VectorStore
from app.knowledge.rag_service import RAGService


def parse_args():
    parser = argparse.ArgumentParser(description="Reindex University AI Agent documents and recover ChromaDB collection.")
    parser.add_argument("--env", type=str, default="development", choices=["development", "testing", "production"],
                        help="Environment profile to use for settings (default: development).")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Custom path to SQLite database file.")
    parser.add_argument("--chroma-dir", type=str, default=None,
                        help="Custom path to Chroma persistence directory.")
    parser.add_argument("--documents-dir", type=str, default=None,
                        help="Custom path to physical documents directory.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-embedding of all documents.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Inspect current catalog and index status without reindexing.")
    return parser.parse_args()


async def inspect_status(db: DatabaseManager, store: VectorStore) -> None:
    """Print current document catalog and vector store status."""
    catalog_docs = await db.list_documents()
    status = store.get_index_status()
    print("\n--- REINDEX / CATALOG STATUS ---")
    print(f"SQLite DB: {db.db_path}")
    print(f"Chroma Dir: {store.persist_dir}")
    print(f"Index Backend: {status.get('backend')}")
    print(f"Collection: {status.get('collection_name')}")
    print(f"Total Chunks in Index: {status.get('total_chunks')}")
    print(f"Total Documents in Catalog: {len(catalog_docs)}")
    print("-" * 35)
    for d in catalog_docs:
        print(f"  {d['filename']:<30} | {d['status']:<10} | Chunks: {d.get('chunk_count', 0)}")
    print("-" * 35)


async def main_async(args):
    overrides = {}
    if args.db_path:
        p = Path(args.db_path).resolve()
        overrides["SQLITE_PATH"] = p
        overrides["DATA_DIR"] = p.parent
    if getattr(args, "chroma_dir", None):
        overrides["CHROMA_DIR"] = Path(args.chroma_dir).resolve()
    if args.documents_dir:
        overrides["DOCUMENTS_DIR"] = Path(args.documents_dir).resolve()

    settings_obj = get_settings(env_override=args.env, **overrides)
    db = DatabaseManager(settings_obj=settings_obj)
    store = VectorStore(persist_dir=settings_obj.CHROMA_DIR, settings_obj=settings_obj)
    rag = RAGService(store=store, db=db, settings_obj=settings_obj)

    if args.verify_only:
        await inspect_status(db, store)
        return

    print(f"Starting document reindexing in '{args.env}' environment...")
    print(f"  Database : {db.db_path}")
    print(f"  ChromaDir: {store.persist_dir}")
    print(f"  Force    : {args.force}")

    res = await rag.reindex_all(force_re_embed=args.force)

    print("\n[SUCCESS] Reindex completed:")
    print(f"  Total Catalog Documents : {res.get('total_catalog_documents')}")
    print(f"  Successfully Reindexed  : {res.get('reindexed')}")
    print(f"  Missing Physical Files  : {res.get('missing')}")
    print(f"  Failed Documents        : {res.get('failed')}")
    print(f"  Total Chunks Generated  : {res.get('total_chunks')}")

    if res.get("details"):
        print("\nDetails:")
        for item in res["details"]:
            status_str = item.get("status")
            err = f" ({item.get('error')})" if item.get("error") else ""
            chunks = f" [{item.get('chunks')} chunks]" if item.get("chunks") is not None else ""
            print(f"  - {item.get('filename')}: {status_str}{chunks}{err}")

    await inspect_status(db, store)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
