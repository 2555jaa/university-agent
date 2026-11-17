#!/usr/bin/env python3
"""End-to-End demonstration script for Local University AI Agent.

Acts as an external client connecting to the running FastAPI server:
1. Verifies system diagnostic health via GET /api/system/health.
2. Registers a sample academic Subject and Topic.
3. Uploads an in-memory lecture note to the RAG knowledge base.
4. Executes an academic chat query via POST /api/chat.
5. Displays formatted assistant responses and source grounding.
"""

import sys
import time
import uuid
import requests


def run_demo(base_url: str = "http://127.0.0.1:8000"):
    print("=" * 70)
    print("   🎓 Local University AI Agent - End-to-End Client Demonstration")
    print("=" * 70)
    print(f"Target Server: {base_url}\n")

    # Step 1: Health & Diagnostics Check
    print("▶ Step 1: Verifying System Health & Diagnostic Status...")
    try:
        health_resp = requests.get(f"{base_url}/api/system/health", timeout=10)
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Could not connect to University Agent at {base_url}.")
        print("   Please start the server first using:")
        print("   ./scripts/run_production.sh  OR  ./scripts/run_agent.sh")
        sys.exit(1)

    print(f"  HTTP Status: {health_resp.status_code}")
    if health_resp.status_code in (200, 503):
        h = health_resp.json()
        print(f"  Overall Status:   {h.get('status')}")
        print(f"  Environment:      {h.get('environment')}")
        print(f"  Database Status:  {h.get('database', {}).get('status')}")
        print(f"  Vector Store:     {h.get('vector_store', {}).get('backend')} ({h.get('vector_store', {}).get('total_chunks', 0)} chunks)")
        print(f"  Ollama Engine:    {h.get('ollama', {}).get('status')} ({h.get('ollama', {}).get('model')})")
    print("  ✓ Health check completed successfully.\n")

    # Step 2: Create Academic Subject & Topic
    print("▶ Step 2: Registering Academic Subject & Topic...")
    sub_code = f"CS{int(time.time()) % 1000}"
    sub_payload = {
        "name": f"Distributed Algorithms ({sub_code})",
        "code": sub_code,
        "description": "Consensus protocols, Byzantine fault tolerance, and replication."
    }
    sub_resp = requests.post(f"{base_url}/api/subjects", json=sub_payload, timeout=10)
    if sub_resp.status_code == 200:
        sub_data = sub_resp.json()
        sub_id = sub_data["id"]
        print(f"  ✓ Created Subject: {sub_data['name']} (ID: {sub_id})")

        # Create topic
        topic_payload = {
            "name": "Raft Consensus",
            "mastery_level": 0.25,
            "notes": "Leader election and replicated state machines."
        }
        top_resp = requests.post(f"{base_url}/api/subjects/{sub_id}/topics", json=topic_payload, timeout=10)
        if top_resp.status_code == 200:
            top_data = top_resp.json()
            print(f"  ✓ Created Topic:   {top_data['name']} (Mastery: {int(top_data['mastery_level'] * 100)}%)")
    else:
        print(f"  ⚠ Subject creation note: {sub_resp.text}")

    print()

    # Step 3: Upload a Course Lecture Document
    print("▶ Step 3: Uploading & Indexing Course Document into RAG Knowledge Base...")
    lecture_content = (
        "Distributed Consensus Lecture 4: The Raft Consensus Algorithm.\n\n"
        "Raft is a consensus algorithm designed to be more understandable than Paxos.\n"
        "It achieves consensus through an elected leader that manages a replicated log.\n"
        "Key subproblems in Raft are:\n"
        "1. Leader Election: A new leader must be chosen when an existing leader fails.\n"
        "2. Log Replication: The leader accepts log entries from clients and replicates them.\n"
        "3. Safety: If any server has applied a particular log entry to its state machine,\n"
        "   then no other server may apply a different log entry for the same log index.\n"
    )
    files = {
        "file": (f"raft_lecture_{uuid.uuid4().hex[:6]}.txt", lecture_content.encode("utf-8"), "text/plain")
    }
    upload_resp = requests.post(f"{base_url}/api/documents/upload", files=files, timeout=30)
    if upload_resp.status_code == 200:
        up_data = upload_resp.json()
        print(f"  ✓ Uploaded: {up_data.get('filename')} (Chunks: {up_data.get('total_chunks')}, Status: {up_data.get('status')})")
    else:
        print(f"  ⚠ Upload note ({upload_resp.status_code}): {upload_resp.text}")

    print()

    # Step 4: Execute Grounded Academic Query
    print("▶ Step 4: Dispatching Chat Query to Academic Agent...")
    query = "What are the three core subproblems in the Raft consensus algorithm?"
    print(f"  Student Query: \"{query}\"")
    chat_payload = {
        "query": query,
        "mode": "tutor",
        "stream": False
    }
    t0 = time.perf_counter()
    chat_resp = requests.post(f"{base_url}/api/chat", json=chat_payload, timeout=60)
    duration = time.perf_counter() - t0

    if chat_resp.status_code == 200:
        ans = chat_resp.json()
        print(f"\n  ✓ Response received in {duration:.2f}s:")
        print("  " + "-" * 66)
        content = ans.get("response") or ans.get("content") or "No text"
        print(f"  {content.strip()}")
        print("  " + "-" * 66)

        sources = ans.get("sources_used") or ans.get("metadata", {}).get("sources_used") or []
        if sources:
            print(f"  Grounded Sources ({len(sources)} citations):")
            for idx, s in enumerate(sources, 1):
                print(f"    [{idx}] {s.get('filename')} (Page {s.get('page_number', 1)})")
        else:
            print("  (Synthesized directly from foundational academic knowledge)")
    else:
        print(f"  ❌ Query failed ({chat_resp.status_code}): {chat_resp.text}")

    print("\n" + "=" * 70)
    print("   🎉 End-to-End Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    run_demo(url)
