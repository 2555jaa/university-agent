"""CLI utility to ingest documents into the University AI Agent knowledge base."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.knowledge.rag_service import rag_service


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_documents.py <file_or_directory_path> [subject]")
        sys.exit(1)

    target_path = Path(sys.argv[1]).resolve()
    subject = sys.argv[2] if len(sys.argv) > 2 else "General"

    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = [
            f for f in target_path.iterdir()
            if f.suffix.lower() in [".pdf", ".docx", ".doc", ".txt", ".md"]
        ]
    else:
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    print(f"Found {len(files)} document(s) to ingest for subject '{subject}'...")
    for f in files:
        print(f"Ingesting: {f.name}...", end=" ", flush=True)
        try:
            res = await rag_service.ingest_document(f, subject=subject)
            print(f"✓ Done ({res.total_pages} pages, {res.total_chunks} chunks)")
        except Exception as e:
            print(f"✗ Failed: {e}")

    print("\nAll documents processed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
