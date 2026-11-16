"""Realistic University Academic Curriculum Seeder for University AI Agent."""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings, settings as default_settings
from app.memory.database import DatabaseManager, db as default_db

logger = logging.getLogger("university_agent.memory.seeder")

SUBJECTS_SEED_DATA = [
    {
        "id": "CS201",
        "code": "CS201",
        "name": "Operating Systems & Systems Programming",
        "description": "Kernel architecture, process scheduling, concurrency, virtual memory, and file systems."
    },
    {
        "id": "MT204",
        "code": "MT204",
        "name": "Discrete Mathematics & Graph Theory",
        "description": "Graph theory, Eulerian and Hamiltonian paths, propositional logic, and combinatorics."
    },
    {
        "id": "AI301",
        "code": "AI301",
        "name": "Artificial Intelligence & Governance",
        "description": "AI safety, model alignment, algorithmic accountability, ethics, and regulatory compliance."
    },
    {
        "id": "PH102",
        "code": "PH102",
        "name": "University Physics & Fundamental Constants",
        "description": "Classical mechanics, thermodynamics, electromagnetism, and universal physical constants."
    }
]

TOPICS_SEED_DATA = [
    # CS201 Topics
    {
        "id": "top_cs201_vm",
        "subject_id": "CS201",
        "name": "Virtual Memory & Paging",
        "notes": "Address translation, page tables, TLB hits/misses, and multi-level paging.",
        "mastery": 0.85
    },
    {
        "id": "top_cs201_pra",
        "subject_id": "CS201",
        "name": "Page Replacement Algorithms",
        "notes": "FIFO, Optimal (Belady), LRU approximation (clock algorithm), and thrashing.",
        "mastery": 0.75
    },
    {
        "id": "top_cs201_sched",
        "subject_id": "CS201",
        "name": "Process Scheduling",
        "notes": "Preemptive scheduling, Round Robin, CFS, multi-level feedback queues.",
        "mastery": 0.70
    },
    # MT204 Topics
    {
        "id": "top_mt204_euler",
        "subject_id": "MT204",
        "name": "Euler & Hamiltonian Paths",
        "notes": "Degrees of vertices, Königsberg bridge problem, Dirac and Ore theorems.",
        "mastery": 0.90
    },
    {
        "id": "top_mt204_trees",
        "subject_id": "MT204",
        "name": "Trees & Connectivity",
        "notes": "Spanning trees, Kruskal/Prim algorithms, graph bipartite properties.",
        "mastery": 0.65
    },
    {
        "id": "top_mt204_logic",
        "subject_id": "MT204",
        "name": "Propositional Logic",
        "notes": "Truth tables, logical equivalences, resolution theorem proving, normal forms.",
        "mastery": 0.80
    },
    # AI301 Topics
    {
        "id": "top_ai301_safety",
        "subject_id": "AI301",
        "name": "AI Safety & Policy",
        "notes": "Risk classification under EU AI Act, catastrophic risk auditing, red teaming.",
        "mastery": 0.85
    },
    {
        "id": "top_ai301_align",
        "subject_id": "AI301",
        "name": "Model Alignment",
        "notes": "RLHF, DPO, constitutional AI, prompt fencing, and jailbreak prevention.",
        "mastery": 0.80
    },
    {
        "id": "top_ai301_account",
        "subject_id": "AI301",
        "name": "Algorithmic Accountability",
        "notes": "Explainability (SHAP/LIME), bias detection, auditing frameworks, transparency logs.",
        "mastery": 0.75
    },
    # PH102 Topics
    {
        "id": "top_ph102_const",
        "subject_id": "PH102",
        "name": "Physical Constants & SI Units",
        "notes": "Speed of light (c), Planck constant (h), gravitational constant (G), Boltzmann (k).",
        "mastery": 0.95
    },
    {
        "id": "top_ph102_quantum",
        "subject_id": "PH102",
        "name": "Quantum Mechanics Constants",
        "notes": "Reduced Planck constant, fine-structure constant, electron mass, Bohr radius.",
        "mastery": 0.70
    },
    {
        "id": "top_ph102_thermo",
        "subject_id": "PH102",
        "name": "Thermodynamic Laws",
        "notes": "Zeroth, First, Second (entropy), and Third laws of thermodynamics.",
        "mastery": 0.80
    }
]

DOCUMENTS_SEED_DATA = [
    {
        "id": "doc_cs201_vm",
        "filename": "os_virtual_memory.pdf",
        "subject_id": "CS201",
        "title": "Virtual Memory and Paging Architectures",
        "file_type": "pdf",
        "fallback_hash": "7bc7b167ef200f867d4bbc1088fca2c3497634d1026169729d55e52bf7b7c5e2",
        "fallback_size": 18240,
        "page_count": 8,
        "chunk_count": 24
    },
    {
        "id": "doc_mt204_discrete",
        "filename": "discrete_math.txt",
        "subject_id": "MT204",
        "title": "Discrete Mathematics: Graph Theory and Eulerian Circuits",
        "file_type": "txt",
        "fallback_hash": "63ca072e831e7d42fa4da6640fb3bf4b91acf654d2fe2f07f09a1f6ce4c9d381",
        "fallback_size": 70,
        "page_count": 1,
        "chunk_count": 2
    },
    {
        "id": "doc_ai301_gov",
        "filename": "ai_governance.docx",
        "subject_id": "AI301",
        "title": "AI Governance Framework and Risk Classification",
        "file_type": "docx",
        "fallback_hash": "a777aa64f62ae892727331b29b6d658c469e947d51bc4d6a6176979ae60b3b16",
        "fallback_size": 38192,
        "page_count": 12,
        "chunk_count": 36
    },
    {
        "id": "doc_ph102_const",
        "filename": "physics_constants.pdf",
        "subject_id": "PH102",
        "title": "Fundamental Physical Constants and SI Standards",
        "file_type": "pdf",
        "fallback_hash": "32e88ae45666e39778b9d7c382495c6e033ed8f4f680fdd40c951d670ec28e73",
        "fallback_size": 14208,
        "page_count": 4,
        "chunk_count": 12
    }
]

SESSIONS_SEED_DATA = [
    {
        "id": "sess_cs201_paging",
        "session_type": "tutor",
        "title": "Virtual Memory and Paging Concepts",
        "subject_id": "CS201",
        "messages": [
            {
                "id": "msg_cs201_01",
                "role": "user",
                "content": "Can you explain the difference between paging and segmentation in OS memory management?",
                "citations": []
            },
            {
                "id": "msg_cs201_02",
                "role": "assistant",
                "content": "Paging divides memory into fixed-size frames and pages, avoiding external fragmentation. Segmentation divides memory into variable-sized logical segments like code and stack.",
                "citations": [{"filename": "os_virtual_memory.pdf", "page_number": 2}]
            }
        ]
    },
    {
        "id": "sess_mt204_euler",
        "session_type": "research",
        "title": "Eulerian Circuits and Path Theorems",
        "subject_id": "MT204",
        "messages": [
            {
                "id": "msg_mt204_01",
                "role": "user",
                "content": "What is the theorem for determining whether a connected graph has an Euler path?",
                "citations": []
            },
            {
                "id": "msg_mt204_02",
                "role": "assistant",
                "content": "A connected graph has an Euler path if and only if it has exactly zero or two vertices of odd degree.",
                "citations": [{"filename": "discrete_math.txt", "page_number": 1}]
            }
        ]
    }
]

QUIZZES_SEED_DATA = [
    {
        "id": "quiz_cs201_01",
        "topic_id": "top_cs201_vm",
        "question": "What component speeds up virtual-to-physical address translation in the CPU?",
        "student_answer": "Translation Lookaside Buffer (TLB)",
        "correct_answer": "Translation Lookaside Buffer (TLB)",
        "is_correct": True,
        "score": 1.0,
        "feedback": "Correct! The TLB caches recently accessed page table translations.",
        "subject_id": "CS201"
    },
    {
        "id": "quiz_mt204_01",
        "topic_id": "top_mt204_euler",
        "question": "How many vertices of odd degree can an Eulerian circuit have?",
        "student_answer": "0",
        "correct_answer": "0",
        "is_correct": True,
        "score": 1.0,
        "feedback": "Correct! In an Eulerian circuit, all vertices must have even degree.",
        "subject_id": "MT204"
    },
    {
        "id": "quiz_ai301_01",
        "topic_id": "top_ai301_safety",
        "question": "What classification tier applies to biometric identification under the EU AI Act?",
        "student_answer": "High Risk",
        "correct_answer": "High Risk",
        "is_correct": True,
        "score": 0.9,
        "feedback": "Accurate! Real-time remote biometric identification is heavily regulated as high risk.",
        "subject_id": "AI301"
    }
]

TASKS_SEED_DATA = [
    {
        "id": "task_cs201_lab",
        "title": "Lab 3: Implement LRU Page Replacement in C",
        "description": "Simulate page replacement using a circular queue clock algorithm.",
        "subject_id": "CS201",
        "deadline": "2026-11-25T23:59:00",
        "priority": "high",
        "status": "in_progress"
    },
    {
        "id": "task_mt204_ps",
        "title": "Problem Set 4: Graph Connectivity Proofs",
        "description": "Complete exercises 4.1 to 4.5 on Hamiltonian paths and graph colorability.",
        "subject_id": "MT204",
        "deadline": "2026-11-28T17:00:00",
        "priority": "medium",
        "status": "pending"
    },
    {
        "id": "task_ai301_essay",
        "title": "Case Study: European AI Act Compliance Analysis",
        "description": "Analyze compliance requirements for generative AI foundation models.",
        "subject_id": "AI301",
        "deadline": "2026-12-05T23:59:00",
        "priority": "high",
        "status": "pending"
    },
    {
        "id": "task_ph102_pset",
        "title": "Physics Problem Set 2: Fundamental Constants Calculations",
        "description": "Calculate Planck energy and Schwarzschild radius for celestial masses.",
        "subject_id": "PH102",
        "deadline": "2026-11-15T18:00:00",
        "priority": "low",
        "status": "completed"
    }
]

GOALS_SEED_DATA = [
    {
        "id": "goal_cs201_master",
        "title": "Master Virtual Memory & Storage Subsystems",
        "description": "Achieve 85%+ on all OS systems programming lab assignments.",
        "target_date": "2026-12-15",
        "status": "in_progress",
        "progress_percent": 65.0,
        "subject_id": "CS201"
    },
    {
        "id": "goal_mt204_exam",
        "title": "Score 90%+ in Discrete Mathematics Final Exam",
        "description": "Review proofs on graphs, trees, and logic equivalence.",
        "target_date": "2026-12-10",
        "status": "in_progress",
        "progress_percent": 40.0,
        "subject_id": "MT204"
    },
    {
        "id": "goal_ai301_cert",
        "title": "Complete AI Governance and Compliance Case Studies",
        "description": "Present policy recommendations on algorithmic auditing.",
        "target_date": "2026-12-20",
        "status": "in_progress",
        "progress_percent": 80.0,
        "subject_id": "AI301"
    }
]


class AcademicSeeder:
    """Deterministic, idempotent seeder for academic coursework and memory data."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        settings_obj: Optional[Settings] = None
    ):
        self.settings = settings_obj or default_settings
        self.db = db or default_db

    async def _get_table_counts(self) -> Dict[str, int]:
        """Query current record counts for all academic tables."""
        table_map = {
            "subjects": "subjects",
            "topics": "topics",
            "documents": "documents",
            "sessions": "learning_sessions",
            "messages": "messages",
            "quizzes": "quiz_results",
            "tasks": "academic_tasks",
            "goals": "study_goals",
        }
        counts = {}
        async with self.db.get_connection() as conn:
            for key, table in table_map.items():
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                counts[key] = (await cursor.fetchone())[0]
        return counts

    async def seed_academic_workspace(
        self,
        include_sample_activity: bool = True
    ) -> Dict[str, Any]:
        """Idempotently seed subjects, topics, documents, sessions, quizzes, tasks, and goals."""
        # Ensure DB schema is initialized
        self.db.init_db_sync()

        initial_counts = await self._get_table_counts()

        # 1. Seed Subjects
        for sub in SUBJECTS_SEED_DATA:
            await self.db.add_subject(
                subject_id=sub["id"],
                name=sub["name"],
                description=sub["description"],
                code=sub["code"]
            )

        # 2. Seed Topics
        for top in TOPICS_SEED_DATA:
            await self.db.add_or_update_topic(
                topic_id=top["id"],
                subject_id=top["subject_id"],
                name=top["name"],
                mastery=top["mastery"],
                notes=top["notes"]
            )

        # 3. Seed Document Catalog
        docs_dir = self.settings.DOCUMENTS_DIR
        for doc in DOCUMENTS_SEED_DATA:
            phys_path = docs_dir / doc["filename"]
            if phys_path.exists():
                file_bytes = phys_path.read_bytes()
                actual_hash = hashlib.sha256(file_bytes).hexdigest()
                actual_size = len(file_bytes)
                resolved_path = str(phys_path.resolve())
            else:
                actual_hash = doc["fallback_hash"]
                actual_size = doc["fallback_size"]
                resolved_path = str((docs_dir / doc["filename"]).resolve())

            await self.db.upsert_document(
                doc_id=doc["id"],
                filename=doc["filename"],
                file_path=resolved_path,
                file_hash=actual_hash,
                file_type=doc["file_type"],
                file_size_bytes=actual_size,
                subject_id=doc["subject_id"],
                title=doc["title"],
                page_count=doc["page_count"],
                chunk_count=doc["chunk_count"],
                status="indexed"
            )

        # 4. Seed Sample Activity (Optional)
        if include_sample_activity:
            # Sessions & Messages
            for sess in SESSIONS_SEED_DATA:
                await self.db.create_session(
                    session_id=sess["id"],
                    session_type=sess["session_type"],
                    title=sess["title"],
                    subject_id=sess["subject_id"]
                )

                for msg in sess["messages"]:
                    await self.db.save_message(
                        message_id=msg["id"],
                        session_id=sess["id"],
                        role=msg["role"],
                        content=msg["content"],
                        mode=sess["session_type"],
                        citations=msg["citations"]
                    )

            # Quizzes
            for q in QUIZZES_SEED_DATA:
                await self.db.record_quiz_result(
                    result_id=q["id"],
                    topic_id=q["topic_id"],
                    question=q["question"],
                    student_answer=q["student_answer"],
                    correct_answer=q["correct_answer"],
                    is_correct=q["is_correct"],
                    score=q["score"],
                    feedback=q["feedback"],
                    subject_id=q["subject_id"]
                )

            # Tasks
            for t in TASKS_SEED_DATA:
                await self.db.add_task(
                    task_id=t["id"],
                    title=t["title"],
                    description=t["description"],
                    subject_id=t["subject_id"],
                    deadline=t["deadline"],
                    priority=t["priority"]
                )
                if t.get("status") and t["status"] != "pending":
                    await self.db.update_task_status(t["id"], t["status"])

            # Goals
            for g in GOALS_SEED_DATA:
                await self.db.add_study_goal(
                    goal_id=g["id"],
                    title=g["title"],
                    description=g["description"],
                    target_date=g["target_date"],
                    status=g["status"],
                    progress_percent=g["progress_percent"],
                    subject_id=g["subject_id"]
                )

        final_counts = await self._get_table_counts()
        created_counts = {k: final_counts[k] - initial_counts.get(k, 0) for k in final_counts}
        existing_counts = {k: initial_counts.get(k, 0) for k in final_counts}

        summary = {
            "status": "success",
            "counts": final_counts,
            "total": final_counts,
            "created": created_counts,
            "existing": existing_counts
        }

        logger.info(f"Academic workspace seeding completed: {summary}")
        return summary

    def seed_academic_workspace_sync(
        self,
        include_sample_activity: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for seed_academic_workspace."""
        return asyncio.run(self.seed_academic_workspace(include_sample_activity=include_sample_activity))
