"""Unit tests for SQLite persistent memory and academic progress tracking."""

import pytest
from app.memory.database import DatabaseManager


@pytest.fixture
def memory_db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    return DatabaseManager(db_path=db_file)


@pytest.mark.asyncio
async def test_subjects_and_topics(memory_db):
    """Test creating subjects, topics, and calculating mastery."""
    await memory_db.add_subject("cs101", "Computer Science 101", "Intro to CS")
    subjects = await memory_db.list_subjects()
    assert len(subjects) == 1
    assert subjects[0]["name"] == "Computer Science 101"

    await memory_db.add_or_update_topic("t1", "cs101", "Data Structures", mastery=0.5)
    topics = await memory_db.list_topics("cs101")
    assert len(topics) == 1
    assert topics[0]["mastery_level"] == 0.5


@pytest.mark.asyncio
async def test_session_history_persistence(memory_db):
    """Test saving and retrieving conversation messages with citations."""
    session_id = "sess_001"
    await memory_db.create_session(session_id, session_type="tutor", title="Calculus 1")

    # Save user message
    await memory_db.save_message("m1", session_id, "user", "What is the derivative of sin(x)?", mode="tutor")

    # Save assistant message with citation
    citations = [{"filename": "Calculus.pdf", "page_number": 42}]
    await memory_db.save_message("m2", session_id, "assistant", "The derivative is cos(x).", mode="tutor", citations=citations)

    history = await memory_db.get_session_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "What is the derivative of sin(x)?"
    assert history[1]["citations"] == citations


@pytest.mark.asyncio
async def test_quiz_and_progress_tracking(memory_db):
    """Test recording quiz results and automatic topic mastery aggregation."""
    await memory_db.add_subject("s1", "Biology")
    await memory_db.add_or_update_topic("top_bio", "s1", "Cell Biology", mastery=0.0)

    # Student takes a quiz
    await memory_db.record_quiz_result(
        result_id="q1",
        topic_id="top_bio",
        question="What is the powerhouse of the cell?",
        student_answer="Mitochondria",
        correct_answer="Mitochondria",
        is_correct=True,
        score=1.0,
        feedback="Excellent!"
    )

    progress = await memory_db.get_learning_progress()
    assert progress["total_quizzes_taken"] == 1
    assert progress["average_score"] == 1.0
    assert progress["total_correct"] == 1

    # Topic mastery should be automatically updated to 1.0
    topics = await memory_db.list_topics("s1")
    assert topics[0]["mastery_level"] == 1.0


@pytest.mark.asyncio
async def test_academic_task_management(memory_db):
    """Test academic tasks and deadline status updates."""
    task = await memory_db.add_task(
        task_id="task_1",
        title="Complete Linear Algebra Problem Set 3",
        deadline="2026-11-20T23:59:00",
        priority="high"
    )
    assert task["status"] == "pending"

    # Mark completed
    updated = await memory_db.update_task_status("task_1", "completed")
    assert updated is True

    pending_tasks = await memory_db.list_tasks(status="pending")
    assert len(pending_tasks) == 0

    all_tasks = await memory_db.list_tasks()
    assert len(all_tasks) == 1
    assert all_tasks[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_subject_topic_hierarchy_enforcements(memory_db):
    """Verify that topics strictly require a valid, existing subject_id and cannot be orphaned."""
    # 1. Attempting to add topic with empty subject_id raises ValueError
    with pytest.raises(ValueError, match="Subject ID cannot be empty"):
        await memory_db.add_or_update_topic("t1", "", "Algorithms")

    # 2. Attempting to add topic with non-existent subject_id raises ValueError
    with pytest.raises(ValueError, match="does not exist"):
        await memory_db.add_or_update_topic("t1", "nonexistent_subject", "Algorithms")

    # 3. Attempting to fetch topic by name with non-existent subject_id raises ValueError
    with pytest.raises(ValueError, match="does not exist"):
        await memory_db.get_topic_by_name("nonexistent_subject", "Algorithms")

    # 4. Valid subject allows topic creation and retrieval
    await memory_db.add_subject("cs201", "Data Structures & Algorithms")
    created_topic = await memory_db.add_or_update_topic("top_algo", "cs201", "Binary Search Trees", mastery=0.6)
    assert created_topic["subject_id"] == "cs201"
    assert created_topic["name"] == "Binary Search Trees"

    fetched = await memory_db.get_topic_by_name("cs201", "Binary Search Trees")
    assert fetched is not None
    assert fetched["id"] == "top_algo"
    assert fetched["mastery_level"] == 0.6

    topics_list = await memory_db.list_topics("cs201")
    assert len(topics_list) == 1
    assert topics_list[0]["name"] == "Binary Search Trees"


@pytest.mark.asyncio
async def test_save_message_rich_metadata_serialization_and_recent_sessions(memory_db):
    """Verify save_message serializes/deserializes JSON metadata and get_recent_sessions tracks active sessions."""
    sess_id = "sess_meta_1"
    await memory_db.create_session(sess_id, session_type="tutor", title="Machine Learning")

    meta = {
        "used_sources": [{"filename": "ml_notes.pdf", "page_number": 12}],
        "reasoning_path": "Step 1: Check gradient descent learning rate.\nStep 2: Calculate cost.",
        "intent": "TUTORING"
    }

    # Save message using standard (session_id, role, content, metadata=...) signature
    saved = await memory_db.save_message(
        session_id=sess_id,
        role="assistant",
        content="Gradient descent updates parameters iteratively.",
        metadata=meta
    )

    assert saved["metadata"]["reasoning_path"] == meta["reasoning_path"]
    assert saved["metadata"]["intent"] == "TUTORING"
    assert len(saved["metadata"]["used_sources"]) == 1

    # Fetch session history and verify deserialization
    history = await memory_db.get_session_history(sess_id)
    assert len(history) == 1
    msg = history[0]
    assert isinstance(msg["metadata"], dict)
    assert msg["metadata"]["reasoning_path"] == meta["reasoning_path"]
    assert msg["metadata"]["intent"] == "TUTORING"
    assert msg["citations"] == meta["used_sources"]

    # Verify get_recent_sessions returns session in order
    recent = await memory_db.get_recent_sessions(limit=5)
    assert len(recent) >= 1
    assert any(s["id"] == sess_id for s in recent)
