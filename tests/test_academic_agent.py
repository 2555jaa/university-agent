"""Unit and integration tests for academic intent classification, dynamic personas, and reasoning extraction."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from app.agent.orchestrator import (
    AcademicAgent,
    AgentQueryRequest,
    AcademicIntent,
    classify_academic_intent,
    extract_reasoning_path,
    extract_json_payload,
    sanitize_tool_output,
    parse_tool_calls,
)
from app.agent.prompts import (
    TUTORING_PROMPT,
    SUMMARIZATION_PROMPT,
    ASSIGNMENT_GUARDRAIL_PROMPT,
    DEFAULT_RESEARCH_PROMPT,
    QUIZ_GENERATOR_PROMPT,
)
from app.knowledge.rag_service import GroundedContext, ContextStatus
from app.providers.base import LLMResponse
from app.security.validator import ActionValidator


def test_intent_classification_categorization():
    """Verify that user queries are correctly categorized into TUTORING, SUMMARIZATION, ASSIGNMENT_ASSISTANCE, and DEFAULT_QA."""
    # Tutoring
    assert classify_academic_intent("Explain Euler paths step by step") == AcademicIntent.TUTORING
    assert classify_academic_intent("How does virtual memory paging work?") == AcademicIntent.TUTORING
    assert classify_academic_intent("Walk me through Dijkstra algorithm") == AcademicIntent.TUTORING
    assert classify_academic_intent("I don't understand how recursion works") == AcademicIntent.TUTORING
    assert classify_academic_intent("Teach me dynamic programming") == AcademicIntent.TUTORING

    # Summarization
    assert classify_academic_intent("Summarize Chapter 4") == AcademicIntent.SUMMARIZATION
    assert classify_academic_intent("Give me the key points of the reading") == AcademicIntent.SUMMARIZATION
    assert classify_academic_intent("Overview of operating system processes") == AcademicIntent.SUMMARIZATION
    assert classify_academic_intent("TL;DR of the EU AI Act") == AcademicIntent.SUMMARIZATION
    assert classify_academic_intent("Provide a brief summary and main takeaways") == AcademicIntent.SUMMARIZATION

    # Assignment Assistance
    assert classify_academic_intent("Write this essay for me on climate change") == AcademicIntent.ASSIGNMENT_ASSISTANCE
    assert classify_academic_intent("Solve this homework problem on eigenvalues") == AcademicIntent.ASSIGNMENT_ASSISTANCE
    assert classify_academic_intent("Help me with my assignment on graph theory") == AcademicIntent.ASSIGNMENT_ASSISTANCE
    assert classify_academic_intent("Do my homework for CS 101") == AcademicIntent.ASSIGNMENT_ASSISTANCE
    assert classify_academic_intent("Give me the answers to the assignment question") == AcademicIntent.ASSIGNMENT_ASSISTANCE

    # Default Q&A
    assert classify_academic_intent("What is the speed of light in vacuum?") == AcademicIntent.DEFAULT_QA
    assert classify_academic_intent("Who formulated Newton's laws?") == AcademicIntent.DEFAULT_QA
    assert classify_academic_intent("List the capital cities of Scandinavia") == AcademicIntent.DEFAULT_QA


@pytest.mark.asyncio
async def test_assignment_keywords_trigger_assignment_guardrail_prompt():
    """Verify that queries matching assignment keywords trigger the ASSIGNMENT_GUARDRAIL_PROMPT."""
    agent = AcademicAgent()

    with patch.object(agent.router, "route_and_generate") as mock_route:
        mock_route.return_value = LLMResponse(
            content="I cannot write your essay directly, but here is a structural outline and framework.",
            model="qwen2.5:1.5b",
            provider="ollama"
        )

        req = AgentQueryRequest(message="Write this essay for me on ethics in computing", use_rag=False)
        res = await agent.process_query(req)

        # Inspect system prompt sent to model router
        sent_messages = mock_route.call_args.kwargs["messages"]
        system_msg = next(m.content for m in sent_messages if m.role == "system")

        assert "ASSIGNMENT & HOMEWORK ASSISTANCE (Academic Integrity Guardrail)" in system_msg
        assert "STRICT ACADEMIC INTEGRITY ENFORCEMENT" in system_msg
        assert "EXPLICITLY FORBIDDEN from writing full essays" in system_msg
        assert res.intent == "ASSIGNMENT_ASSISTANCE"


@pytest.mark.asyncio
async def test_think_tags_parsed_into_reasoning_path_and_stripped_from_reply():
    """Verify that <think>...</think> tags in LLM output are captured in reasoning_path and stripped from user reply."""
    agent = AcademicAgent()

    simulated_llm_output = (
        "<think>\n"
        "Step 1: Check vertex degrees in the input graph.\n"
        "Step 2: Two vertices have odd degrees (degree=3 and degree=1).\n"
        "Step 3: Conclude that an Euler path exists but not an Euler circuit.\n"
        "</think>\n"
        "An Euler path exists because exactly two vertices have odd degree [1]."
    )

    with patch.object(agent.router, "route_and_generate") as mock_route:
        mock_route.return_value = LLMResponse(
            content=simulated_llm_output,
            model="deepseek-r1:latest",
            provider="ollama"
        )

        req = AgentQueryRequest(message="Can you explain if an Euler path exists?", use_rag=False)
        res = await agent.process_query(req)

        # 1. <think> tags stripped from user-facing reply
        assert "<think>" not in res.reply
        assert "</think>" not in res.reply
        assert "Step 1: Check vertex degrees" not in res.reply
        assert "An Euler path exists because exactly two vertices have odd degree" in res.reply

        # 2. Reasoning captured in dedicated field
        assert res.reasoning_path is not None
        assert "Step 1: Check vertex degrees in the input graph." in res.reasoning_path
        assert "Step 3: Conclude that an Euler path exists" in res.reasoning_path


@pytest.mark.asyncio
async def test_dynamic_persona_swapping_summarization_and_tutoring():
    """Verify dynamic system prompt swapping between SUMMARIZATION, TUTORING, and DEFAULT_QA."""
    agent = AcademicAgent()

    with patch.object(agent.router, "route_and_generate") as mock_route:
        mock_route.return_value = LLMResponse(content="Response", model="test", provider="ollama")

        # 1. Summarization intent
        req_sum = AgentQueryRequest(message="Summarize the key points of OS memory management", use_rag=False)
        res_sum = await agent.process_query(req_sum)
        sys_sum = next(m.content for m in mock_route.call_args.kwargs["messages"] if m.role == "system")
        assert "ACADEMIC SUMMARIZATION & DISTILLATION" in sys_sum
        assert res_sum.intent == "SUMMARIZATION"

        # 2. Tutoring intent
        req_tut = AgentQueryRequest(message="Walk me through how virtual memory translates addresses step-by-step", use_rag=False)
        res_tut = await agent.process_query(req_tut)
        sys_tut = next(m.content for m in mock_route.call_args.kwargs["messages"] if m.role == "system")
        assert "UNIVERSITY TUTOR (Socratic Mentorship)" in sys_tut
        assert res_tut.intent == "TUTORING"

        # 3. Default Q&A
        req_qa = AgentQueryRequest(message="What is the speed of light in vacuum?", use_rag=False)
        res_qa = await agent.process_query(req_qa)
        sys_qa = next(m.content for m in mock_route.call_args.kwargs["messages"] if m.role == "system")
        assert "ACADEMIC RESEARCH ASSISTANT (Default Q&A)" in sys_qa
        assert res_qa.intent == "DEFAULT_QA"


def test_unclosed_think_tag_safely_extracted():
    """Verify that an unclosed <think> tag from truncated LLM output is safely parsed."""
    truncated_output = "<think>Drafting intermediate theorem proof without end"
    cleaned, reasoning = extract_reasoning_path(truncated_output)
    assert cleaned == ""
    assert reasoning == "Drafting intermediate theorem proof without end"


def test_extract_json_payload_resilience():
    """Verify extract_json_payload handles fences, conversational text, and reasoning blocks."""
    # 1. Standard markdown fence
    fenced = "```json\n{\"topic\": \"OS\", \"questions\": []}\n```"
    assert extract_json_payload(fenced) == {"topic": "OS", "questions": []}

    # 2. Conversational preamble and postamble
    surrounded = "Here is your requested quiz:\n```json\n{\"topic\": \"Algorithms\", \"questions\": []}\n```\nHope this helps!"
    assert extract_json_payload(surrounded) == {"topic": "Algorithms", "questions": []}

    # 3. Outer brace extraction without markdown fence
    raw_braces = "Sure! {\"topic\": \"Networks\", \"questions\": []} done."
    assert extract_json_payload(raw_braces) == {"topic": "Networks", "questions": []}

    # 4. Embedded <think> reasoning block
    with_think = "<think>Reasoning through questions...</think>\n```json\n{\"topic\": \"Databases\", \"questions\": []}\n```"
    assert extract_json_payload(with_think) == {"topic": "Databases", "questions": []}

    # 5. Invalid JSON throws ValueError
    with pytest.raises(ValueError):
        extract_json_payload("No JSON here whatsoever!")


@pytest.mark.asyncio
async def test_generate_quiz_parses_markdown_fenced_json():
    """Verify generate_quiz parses markdown-fenced JSON and validates source citations."""
    agent = AcademicAgent()

    mock_context = GroundedContext(
        context_text="Virtual memory decouples logical addresses from physical RAM using page tables [1].",
        citations=[{
            "index": 1,
            "chunk_id": "chunk-os-1",
            "filename": "os_virtual_memory.pdf",
            "page_number": 3,
            "heading": "Paging Mechanisms"
        }],
        has_relevant_info=True,
        status=ContextStatus.FOUND
    )

    fenced_llm_quiz = """
    ```json
    {
      "topic": "Virtual Memory",
      "questions": [
        {
          "question_text": "What is the primary role of page tables in virtual memory?",
          "options": [
            "A) Translate virtual addresses to physical frames",
            "B) Manage disk file systems",
            "C) Direct network packets",
            "D) Compile source code"
          ],
          "correct_answer": "A) Translate virtual addresses to physical frames",
          "explanation": "Page tables map logical page numbers to physical page frames.",
          "source_chunk_id": "[1]"
        }
      ]
    }
    ```
    """

    with patch.object(agent.rag, "retrieve_grounded_context", new_callable=AsyncMock) as mock_rag, \
         patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route:
        mock_rag.return_value = mock_context
        mock_route.return_value = LLMResponse(content=fenced_llm_quiz, model="qwen2.5:1.5b", provider="ollama")

        quiz = await agent.generate_quiz(subject_id="cs_os", topic="Virtual Memory", num_questions=1)

        # Verify router received QUIZ_GENERATOR_PROMPT
        sent_messages = mock_route.call_args.kwargs["messages"]
        system_msg = next(m.content for m in sent_messages if m.role == "system")
        assert "ACADEMIC QUIZ GENERATOR" in system_msg
        assert "Required JSON Schema" in system_msg

        # Verify parsed quiz payload
        assert quiz["status"] == "success"
        assert quiz["topic"] == "Virtual Memory"
        assert len(quiz["questions"]) == 1

        q0 = quiz["questions"][0]
        assert q0["question_text"] == "What is the primary role of page tables in virtual memory?"
        assert q0["correct_answer"] == "A) Translate virtual addresses to physical frames"
        assert q0["source_chunk_id"] == "[1]"
        assert q0["source_metadata"] is not None
        assert q0["source_metadata"]["filename"] == "os_virtual_memory.pdf"
        assert q0["source_metadata"]["page_number"] == 3


@pytest.mark.asyncio
async def test_generate_quiz_insufficient_evidence_returns_clear_error():
    """Verify generate_quiz returns a clear error dictionary when context is INSUFFICIENT_EVIDENCE."""
    agent = AcademicAgent()

    insufficient_ctx = GroundedContext(
        context_text="I don't have enough information in your notes to answer this.",
        citations=[],
        has_relevant_info=False,
        status=ContextStatus.INSUFFICIENT_EVIDENCE
    )

    with patch.object(agent.rag, "retrieve_grounded_context", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = insufficient_ctx

        quiz = await agent.generate_quiz(subject_id="physics", topic="Superstring Theory", num_questions=3)

        assert isinstance(quiz, dict)
        assert quiz["status"] == "insufficient_evidence"
        assert "error" in quiz
        assert "Superstring Theory" in quiz["error"]
        assert quiz["questions"] == []


@pytest.mark.asyncio
async def test_generate_quiz_unparseable_json_returns_safe_error_dict():
    """Verify generate_quiz returns a safe error dictionary if the LLM output is entirely unparseable."""
    agent = AcademicAgent()

    mock_context = GroundedContext(
        context_text="Discrete math deals with countable sets.",
        citations=[{"index": 1, "chunk_id": "c1", "filename": "discrete_math.txt"}],
        has_relevant_info=True,
        status=ContextStatus.FOUND
    )

    with patch.object(agent.rag, "retrieve_grounded_context", new_callable=AsyncMock) as mock_rag, \
         patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route:
        mock_rag.return_value = mock_context
        # LLM outputs non-JSON plain text
        mock_route.return_value = LLMResponse(
            content="I am sorry, but as an AI assistant I cannot generate quizzes today.",
            model="qwen2.5:1.5b",
            provider="ollama"
        )

        quiz = await agent.generate_quiz(subject_id="math", topic="Graph Theory", num_questions=2)

        assert isinstance(quiz, dict)
        assert quiz["status"] == "format_error"
        assert "error" in quiz
        assert "Failed to parse structured quiz" in quiz["error"]
        assert quiz["questions"] == []


@pytest.mark.asyncio
async def test_generate_quiz_academic_integrity_hides_answers_when_requested():
    """Verify that hide_answers=True omits or masks correct_answer and explanation for student self-testing."""
    agent = AcademicAgent()

    mock_context = GroundedContext(
        context_text="Binary search runs in O(log n) time.",
        citations=[{"index": 1, "chunk_id": "c1", "filename": "algo.pdf"}],
        has_relevant_info=True,
        status=ContextStatus.FOUND
    )

    raw_quiz = """
    {
      "topic": "Algorithms",
      "questions": [
        {
          "question_text": "What is the worst-case time complexity of binary search?",
          "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n^2)"],
          "correct_answer": "B) O(log n)",
          "explanation": "Binary search divides the search space in half each iteration.",
          "source_chunk_id": "[1]"
        }
      ]
    }
    """

    with patch.object(agent.rag, "retrieve_grounded_context", new_callable=AsyncMock) as mock_rag, \
         patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route:
        mock_rag.return_value = mock_context
        mock_route.return_value = LLMResponse(content=raw_quiz, model="test", provider="ollama")

        quiz = await agent.generate_quiz(topic="Binary Search", num_questions=1, hide_answers=True)

        assert quiz["status"] == "success"
        assert quiz["answers_hidden"] is True
        assert len(quiz["questions"]) == 1
        assert quiz["questions"][0]["correct_answer"] is None
        assert quiz["questions"][0]["explanation"] is None
        assert quiz["questions"][0]["source_chunk_id"] == "[1]"


@pytest.mark.asyncio
async def test_process_query_prunes_long_chat_history_preserving_newest_and_system(tmp_path):
    """Verify process_query prunes older chat turns when history is long, keeping newest and system intact."""
    from app.memory.database import DatabaseManager
    test_db = DatabaseManager(db_path=tmp_path / "agent_prune_test.db")
    agent = AcademicAgent(db_instance=test_db)

    sess_id = "sess_long_conversation"
    await test_db.create_session(sess_id, session_type="tutor", title="Long Study Session")

    # Seed 20 historical messages (each ~40 words / 200 chars ~ 50 tokens)
    for i in range(1, 21):
        role = "user" if i % 2 == 1 else "assistant"
        await test_db.save_message(
            session_id=sess_id,
            role=role,
            content=f"Message {i}: This is an in-depth academic discussion point regarding advanced algorithm analysis and complexity theory part {i}."
        )

    # Set MAX_CONTEXT_TOKENS to 1100 so that 20 long messages are pruned down to recent turns
    agent.settings.MAX_CONTEXT_TOKENS = 1100

    with patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = LLMResponse(content="Final reply", model="test", provider="ollama")

        req = AgentQueryRequest(
            session_id=sess_id,
            message="What is the final conclusion of our analysis?",
            use_rag=False,
            use_tools=False
        )
        res = await agent.process_query(req)

        sent_messages = mock_route.call_args.kwargs["messages"]

        # 1. System prompt is first and preserved intact
        assert sent_messages[0].role == "system"

        # 2. Current user message is last and preserved intact
        assert sent_messages[-1].role == "user"
        assert sent_messages[-1].content == "What is the final conclusion of our analysis?"

        # 3. Message count is pruned: far fewer than 22 messages (1 system + 20 history + 1 current)
        assert len(sent_messages) < 22

        # 4. The newest history messages (e.g. Message 20, 19) are kept, oldest (e.g. Message 1, 2) pruned
        sent_contents = [m.content for m in sent_messages]
        assert any("Message 20:" in c for c in sent_contents)
        assert not any("Message 1:" in c for c in sent_contents)


def test_prune_chat_history_unit():
    """Unit test for prune_chat_history sliding window token estimator."""
    from app.agent.orchestrator import prune_chat_history

    system_prompt = "You are an expert tutor."
    current_message = "Current question?"
    history = [
        {"role": "user", "content": f"Turn {i} " + ("x" * 100)} for i in range(1, 21)
    ]

    # With a small max_context_tokens of 200, only newest turns should be kept
    messages = prune_chat_history(
        history=history,
        system_prompt=system_prompt,
        current_message=current_message,
        max_context_tokens=200,
        reserve_tokens_for_generation=50
    )

    assert messages[0].content == system_prompt
    assert messages[-1].content == current_message
    assert len(messages) < len(history) + 2
    contents = [m.content for m in messages]
    assert any("Turn 20" in c for c in contents)
    assert not any("Turn 1 " in c for c in contents)


@pytest.mark.asyncio
async def test_evaluate_quiz_accurately_scores_four_questions(tmp_path):
    """Verify evaluate_quiz correctly scores a simulated 4-question quiz (3 correct = 0.75) and updates mastery."""
    from app.memory.database import DatabaseManager
    test_db = DatabaseManager(db_path=tmp_path / "test_eval_quiz.db")
    agent = AcademicAgent(db_instance=test_db)

    await test_db.add_subject("cs_sys", "Computer Systems")
    await test_db.add_or_update_topic("top_os_concepts", "cs_sys", "Core OS Concepts", mastery=0.5)

    quiz_data = {
        "topic": "Core OS Concepts",
        "questions": [
            {
                "question_text": "What causes a page fault?",
                "options": ["A) Accessing an unmapped virtual address", "B) Disk failure", "C) CPU overheating", "D) Network disconnect"],
                "correct_answer": "A) Accessing an unmapped virtual address",
                "explanation": "Page faults happen when a referenced page is not in physical memory.",
                "source_chunk_id": "[1]"
            },
            {
                "question_text": "Which scheduling algorithm is non-preemptive?",
                "options": ["A) Round Robin", "B) First-Come First-Served (FCFS)", "C) Shortest Remaining Time First", "D) Priority Preemptive"],
                "correct_answer": "B) First-Come First-Served (FCFS)",
                "explanation": "FCFS processes requests strictly in order of arrival without preemption.",
                "source_chunk_id": "[2]"
            },
            {
                "question_text": "What is a condition required for deadlock?",
                "options": ["A) Mutual Exclusion", "B) Preemption allowed", "C) No resource sharing", "D) Single thread"],
                "correct_answer": "A) Mutual Exclusion",
                "explanation": "Deadlock requires mutual exclusion, hold and wait, no preemption, and circular wait.",
                "source_chunk_id": "[3]"
            },
            {
                "question_text": "What synchronization primitive prevents race conditions?",
                "options": ["A) Semaphore", "B) Cache miss", "C) Page table", "D) TLB buffer"],
                "correct_answer": "A) Semaphore",
                "explanation": "Semaphores provide mutual exclusion to shared critical sections.",
                "source_chunk_id": "[4]"
            }
        ]
    }

    # Student chooses 3 correct answers and 1 incorrect answer
    student_answers = {
        "0": "A) Accessing an unmapped virtual address",  # Correct
        "1": "B",                                         # Correct (option letter)
        "2": "A) Mutual Exclusion",                       # Correct
        "3": "C) Page table"                              # Incorrect (correct is A)
    }

    res = await agent.evaluate_quiz(
        topic_id="top_os_concepts",
        answers=student_answers,
        quiz_data=quiz_data
    )

    assert res["score"] == 0.75
    assert res["total_questions"] == 4
    assert res["correct_count"] == 3
    assert len(res["evaluations"]) == 4

    assert res["evaluations"][0]["is_correct"] is True
    assert res["evaluations"][1]["is_correct"] is True
    assert res["evaluations"][2]["is_correct"] is True
    assert res["evaluations"][3]["is_correct"] is False

    # Check topic mastery was updated via EMA: (0.5 * 0.7) + (0.75 * 0.3) = 0.35 + 0.225 = 0.575
    topic = await test_db.get_topic_by_name("cs_sys", "Core OS Concepts")
    assert topic["mastery_level"] == 0.575


@pytest.mark.asyncio
async def test_sanitize_tool_output_truncates_large_payload():
    """Verify sanitize_tool_output safely truncates large tool outputs while leaving short outputs intact."""
    short_text = "Operation completed successfully."
    assert sanitize_tool_output(short_text, max_length=1500) == short_text

    # Dict serialization
    data = {"status": "success", "count": 10}
    sanitized_dict = sanitize_tool_output(data, max_length=1500)
    assert "success" in sanitized_dict

    # 5,000 character output
    large_payload = "A" * 5000
    sanitized_large = sanitize_tool_output(large_payload, max_length=1500)
    assert len(sanitized_large) <= 1500
    assert "...[OUTPUT TRUNCATED DUE TO SIZE]..." in sanitized_large
    assert sanitized_large.startswith("AAAA")
    assert sanitized_large.endswith("AAAA")


def test_parse_tool_calls_formats():
    """Verify tool call parser supports XML, attributes, and JSON code block formats."""
    # 1. Standard XML
    xml_call = '<tool_call>\n{"name": "read_file", "arguments": {"path": "test.txt"}}\n</tool_call>'
    parsed = parse_tool_calls(xml_call)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "read_file"
    assert parsed[0]["arguments"] == {"path": "test.txt"}

    # 2. Alternate XML <tool>
    alt_xml = '<tool>\n{"name": "list_directory", "arguments": {"path": "src"}}\n</tool>'
    parsed = parse_tool_calls(alt_xml)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "list_directory"
    assert parsed[0]["arguments"] == {"path": "src"}

    # 3. Attribute tag
    attr_call = '<tool name="read_file">{"path": "config.json"}</tool>'
    parsed = parse_tool_calls(attr_call)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "read_file"
    assert parsed[0]["arguments"] == {"path": "config.json"}

    # 4. JSON markdown code block
    json_block = '```json\n{"name": "run_safe_command", "arguments": {"command": "ls"}}\n```'
    parsed = parse_tool_calls(json_block)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "run_safe_command"
    assert parsed[0]["arguments"] == {"command": "ls"}


@pytest.mark.asyncio
async def test_orchestrator_tool_call_execution_and_followup(tmp_path):
    """Verify orchestrator parses tool calls, executes the tool, appends role='tool', and triggers follow-up LLM generation."""
    from app.memory.database import DatabaseManager
    test_db = DatabaseManager(db_path=tmp_path / "agent_tool_test.db")
    agent = AcademicAgent(db_instance=test_db)

    # Mock tool call output from LLM in turn 1, and final grounded answer in turn 2
    turn_1_response = LLMResponse(
        content='<tool_call>\n{"name": "read_file", "arguments": {"path": "syllabus.txt"}}\n</tool_call>',
        model="test-model",
        provider="ollama"
    )
    turn_2_response = LLMResponse(
        content="Based on syllabus.txt, Midterm Exam 1 is scheduled for October 24th.",
        model="test-model",
        provider="ollama"
    )

    with patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route, \
         patch.object(agent, "execute_tool", new_callable=AsyncMock) as mock_exec:

        mock_route.side_effect = [turn_1_response, turn_2_response]
        mock_exec.return_value = {
            "status": "success",
            "path": "syllabus.txt",
            "content": "Midterm Exam 1: October 24th. Final Exam: December 15th."
        }

        req = AgentQueryRequest(
            message="When is the midterm exam?",
            use_rag=False,
            use_tools=True
        )

        res = await agent.process_query(req)

        # 1. execute_tool was called for read_file
        mock_exec.assert_awaited_once_with("read_file", {"path": "syllabus.txt"})

        # 2. route_and_generate was called twice (initial turn + follow-up after tool execution)
        assert mock_route.call_count == 2

        # 3. Verify the messages passed into the second route_and_generate call
        follow_up_messages = mock_route.call_args_list[1].kwargs["messages"]
        roles = [m.role for m in follow_up_messages]
        assert "assistant" in roles
        assert "tool" in roles

        tool_msg = next(m for m in follow_up_messages if m.role == "tool")
        assert "Midterm Exam 1" in tool_msg.content

        # 4. Final AgentResponse validation
        assert res.status == "success"
        assert res.reply == "Based on syllabus.txt, Midterm Exam 1 is scheduled for October 24th."
        assert res.tool_calls is not None
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["name"] == "read_file"
        assert res.tool_calls[0]["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_orchestrator_halts_on_approval_required_tool(tmp_path):
    """Verify orchestrator halts execution when a side-effect tool requires approval and no token is provided."""
    from app.memory.database import DatabaseManager
    test_db = DatabaseManager(db_path=tmp_path / "agent_approval_test.db")
    agent = AcademicAgent(db_instance=test_db)

    safe_target = str(agent.settings.DOCUMENTS_DIR / "solution.txt")
    tool_call_text = (
        f'<tool_call>\n'
        f'{{"name": "write_file", "arguments": {{"path": "{safe_target}", "content": "answers"}}}}\n'
        f'</tool_call>'
    )
    llm_res = LLMResponse(content=tool_call_text, model="test-model", provider="ollama")

    with patch.object(agent.router, "route_and_generate", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = llm_res

        req = AgentQueryRequest(
            message="Write the exam answers to solution.txt",
            use_rag=False,
            use_tools=True
        )

        res = await agent.process_query(req)

        # 1. route_and_generate was called only once (execution halted, no follow-up generation)
        assert mock_route.call_count == 1

        # 2. Response status is approval_required
        assert res.status == "approval_required"
        assert res.approval_required is not None
        assert res.approval_required.get("status") == "approval_required"
        assert "approval_id" in res.approval_required
        assert isinstance(res.approval_required["approval_id"], str) and len(res.approval_required["approval_id"]) > 5

        # 3. Tool call record was tracked
        assert res.tool_calls is not None
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["name"] == "write_file"


@pytest.mark.asyncio
async def test_orchestrator_execute_code_approval_and_execution():
    """Verify execute_code tool routes through approval check and sandbox execution."""
    agent = AcademicAgent()

    # Case 1: Without token -> approval_required
    res_no_token = await agent.execute_tool(
        name="execute_code",
        arguments={"code": "print(2 + 2)", "language": "python"}
    )
    assert res_no_token["status"] == "approval_required"
    assert res_no_token["action"] == "execute_code"
    token = res_no_token["approval_id"]

    # Approve token
    ActionValidator.approve_action(token)

    # Case 2: With valid token -> executes in sandbox
    res_with_token = await agent.execute_tool(
        name="execute_code",
        arguments={"code": "print(2 + 2)", "language": "python", "approval_token": token}
    )
    assert res_with_token["status"] == "success"
    assert "4" in res_with_token["stdout"]

