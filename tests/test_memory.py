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


@pytest.mark.asyncio
async def test_record_quiz_result_ema_and_last_reviewed(memory_db):
    """Verify record_quiz_result updates topic mastery via EMA and bumps last_reviewed."""
    await memory_db.add_subject("subj_os", "Operating Systems")
    await memory_db.add_or_update_topic("top_sync", "subj_os", "Process Synchronization", mastery=0.5)

    # First quiz with score 1.0: (0.5 * 0.7) + (1.0 * 0.3) = 0.35 + 0.30 = 0.65
    res1 = await memory_db.record_quiz_result(
        subject_id="subj_os",
        topic_id="top_sync",
        score=1.0,
        total_questions=5
    )
    assert res1["mastery_level"] == 0.65

    topics = await memory_db.list_topics("subj_os")
    assert len(topics) == 1
    assert topics[0]["mastery_level"] == 0.65
    assert topics[0]["last_reviewed"] is not None

    # Second quiz with score 0.5: (0.65 * 0.7) + (0.5 * 0.3) = 0.455 + 0.15 = 0.605
    res2 = await memory_db.record_quiz_result(
        subject_id="subj_os",
        topic_id="top_sync",
        score=0.5,
        total_questions=4
    )
    assert res2["mastery_level"] == 0.605

    topics2 = await memory_db.list_topics("subj_os")
    assert topics2[0]["mastery_level"] == 0.605


@pytest.mark.asyncio
async def test_upcoming_and_overdue_tasks_filtering(memory_db):
    """Verify get_upcoming_tasks and get_overdue_tasks filter by status and chronological deadlines."""
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    t_tomorrow = (now + timedelta(days=1)).isoformat()
    t_in_three_days = (now + timedelta(days=3)).isoformat()
    t_in_twenty_days = (now + timedelta(days=20)).isoformat()
    t_yesterday = (now - timedelta(days=1)).isoformat()

    # 1. Upcoming task due tomorrow (pending)
    await memory_db.add_task("task_due_tomorrow", "Review Lecture 3", deadline=t_tomorrow)

    # 2. Upcoming task due in 3 days (in_progress)
    await memory_db.add_task("task_due_3days", "Lab 2 submission", deadline=t_in_three_days)
    await memory_db.update_task_status("task_due_3days", "in_progress")

    # 3. Completed task due tomorrow (should be excluded from upcoming)
    await memory_db.add_task("task_completed_tomorrow", "Quiz 1 Prep", deadline=t_tomorrow)
    await memory_db.update_task_status("task_completed_tomorrow", "completed")

    # 4. Distant task due in 20 days (should be excluded from 7-day window)
    await memory_db.add_task("task_distant", "Final Exam Study", deadline=t_in_twenty_days)

    # 5. Overdue task due yesterday (pending)
    await memory_db.add_task("task_overdue", "Homework 1", deadline=t_yesterday)

    # Query upcoming tasks within 7 days
    upcoming = await memory_db.get_upcoming_tasks(days_ahead=7)
    upcoming_ids = [t["id"] for t in upcoming]

    assert len(upcoming) == 2
    assert upcoming_ids == ["task_due_tomorrow", "task_due_3days"]
    # Verify sorting by closest deadline ascending
    assert upcoming[0]["deadline"] < upcoming[1]["deadline"]

    # Query overdue tasks
    overdue = await memory_db.get_overdue_tasks()
    assert len(overdue) == 1
    assert overdue[0]["id"] == "task_overdue"


@pytest.mark.asyncio
async def test_update_goal_progress_clamping_and_auto_completion(memory_db):
    """Verify update_goal_progress clamps progress percentage and automatically marks goal completed at 100%."""
    goal = await memory_db.add_study_goal(
        goal_id="goal_ml",
        title="Complete Machine Learning Specialization",
        progress_percent=25.0
    )
    assert goal["status"] == "in_progress"
    assert goal["progress_percent"] == 25.0

    # 1. Update progress to 60%
    u1 = await memory_db.update_goal_progress("goal_ml", 60.0)
    assert u1["progress_percent"] == 60.0
    assert u1["status"] == "in_progress"

    # 2. Update progress with out-of-range value (> 100%) -> clamps to 100% and auto-completes
    u2 = await memory_db.update_goal_progress("goal_ml", 140.0)
    assert u2["progress_percent"] == 100.0
    assert u2["status"] == "completed"

    # 3. Verify get_active_goals excludes completed goal
    active_goals = await memory_db.get_active_goals()
    assert not any(g["id"] == "goal_ml" for g in active_goals)
