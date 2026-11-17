"""Unit tests for security validator, approval workflow, and audit logging."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from app.config import settings
from app.security.validator import SecurityValidator, ActionValidator, ActionRisk, ApprovalStatus, security_validator
from app.security.sandbox import execute_code, validate_python_ast
from app.security.audit import AuditLogger
from app.tools.terminal_tools import RestrictedTerminalTools
from app.tools.filesystem_tools import FilesystemTools
from app.services.file_service import FileService, SecurityError


@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def custom_validator(temp_workspace):
    return SecurityValidator(allowed_roots=[temp_workspace])


def test_path_traversal_prevention(custom_validator, temp_workspace):
    """Test that path traversal attempts outside allowed roots are blocked."""
    safe_file = temp_workspace / "lecture1.txt"
    safe_file.write_text("Linear Algebra Notes")

    # Allowed path
    assert custom_validator.is_safe_path(safe_file) is True
    assert custom_validator.validate_path_access(safe_file) == safe_file.resolve()

    # Blocked external paths
    assert custom_validator.is_safe_path("/etc/passwd") is False
    assert custom_validator.is_safe_path(temp_workspace / "../../../etc/shadow") is False

    with pytest.raises(PermissionError):
        custom_validator.validate_path_access("/etc/passwd")


def test_approval_workflow(custom_validator):
    """Test creating, approving, and rejecting dangerous action requests."""
    req = custom_validator.request_approval(
        action="delete_file",
        description="Delete outdated syllabus",
        params={"path": "/data/syllabus.pdf"},
        risk=ActionRisk.HIGH
    )
    assert req.status == ApprovalStatus.PENDING

    # Retrieve pending
    pending = custom_validator.list_pending()
    assert any(r.id == req.id for r in pending)

    # Approve
    approved = custom_validator.approve_action(req.id, approver_note="Verified by student")
    assert approved is True
    assert custom_validator.get_approval(req.id).status == ApprovalStatus.APPROVED


def test_restricted_terminal_whitelisting(temp_workspace):
    """Test that only safe inspection commands succeed and dangerous ones are blocked."""
    # Whitelisted command (pwd)
    res = RestrictedTerminalTools.run_safe_command("pwd")
    assert res["status"] == "success"

    # Blocked command (rm)
    blocked_res = RestrictedTerminalTools.run_safe_command("rm -rf /")
    assert blocked_res["status"] == "blocked"
    assert "whitelist" in blocked_res["error"]


def test_filesystem_delete_requires_approval(temp_workspace):
    """Test that file deletion requires explicit human approval token."""
    from app.security.validator import security_validator
    security_validator.allowed_roots.append(temp_workspace.resolve())

    test_file = temp_workspace / "old_doc.txt"
    test_file.write_text("temporary data")

    # Deleting without approval token must return approval_required
    res = FilesystemTools.delete_file(str(test_file))
    assert res["status"] == "approval_required"
    assert "approval_id" in res
    assert test_file.exists()  # File still exists


def test_audit_logging(tmp_path):
    """Test append-only JSON audit logging."""
    audit_file = tmp_path / "test_audit.log"
    logger = AuditLogger(log_path=audit_file)

    logger.log_event(
        event_type="test_event",
        action="verify_math",
        details={"score": 100},
        actor="student",
        risk_level="low",
        status="success"
    )

    events = logger.read_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["action"] == "verify_math"
    assert events[0]["details"]["score"] == 100


def test_approval_token_action_and_param_binding(temp_workspace):
    """Test that approval token cannot be used for a different action or target, and cannot be replayed."""
    from app.security.validator import security_validator
    security_validator.allowed_roots.append(temp_workspace.resolve())

    file_a = temp_workspace / "file_a.txt"
    file_b = temp_workspace / "file_b.txt"
    file_a.write_text("file A content")
    file_b.write_text("file B content")

    # 1. Request approval to delete file_a
    req_res = FilesystemTools.delete_file(str(file_a))
    app_id = req_res["approval_id"]
    security_validator.approve_action(app_id, "Approved for file A")

    # 2. Attempt to use file_a's approval token to execute terminal command (action mismatch)
    with pytest.raises(PermissionError) as exc_info:
        RestrictedTerminalTools.execute_with_approval("whoami", approval_id=app_id)
    assert "mismatch" in str(exc_info.value).lower()

    # 3. Attempt to use file_a's approval token to delete file_b (param mismatch)
    with pytest.raises(PermissionError) as exc_info:
        FilesystemTools.delete_file(str(file_b), approval_id=app_id)
    assert "mismatch" in str(exc_info.value).lower()

    # 4. Valid deletion of file_a succeeds
    del_res = FilesystemTools.delete_file(str(file_a), approval_id=app_id)
    assert del_res["status"] == "success"
    assert not file_a.exists()

    # 5. Attempt replay of the consumed approval token must be blocked
    file_a_recreated = temp_workspace / "file_a.txt"
    file_a_recreated.write_text("recreated")
    with pytest.raises(PermissionError) as exc_info:
        FilesystemTools.delete_file(str(file_a_recreated), approval_id=app_id)
    assert "consumed" in str(exc_info.value).lower() or "replay" in str(exc_info.value).lower()


def test_sensitive_target_protection(temp_workspace):
    """Test that sensitive configuration files (.env, config.py) are strictly protected."""
    from app.security.validator import security_validator

    assert security_validator.is_sensitive_target(Path(".env")) is True
    assert security_validator.is_sensitive_target(Path("app/config.py")) is True
    assert security_validator.is_safe_path(".env") is False
    assert security_validator.is_safe_path("app/config.py") is False

    with pytest.raises(PermissionError):
        FilesystemTools.read_authorized_file("app/config.py")

    # Safe run cannot access config.py
    res = RestrictedTerminalTools.run_safe_command("head app/config.py")
    assert res["status"] == "blocked"


def test_approved_token_expires_before_consumption(custom_validator):
    req = custom_validator.request_approval("delete_file", "temporary test", {"path": "/tmp/a"})
    assert custom_validator.approve_action(req.id)
    req.requested_at = time.time() - settings.ACTION_TIMEOUT_SECONDS - 1
    ok, reason = custom_validator.consume_approval(req.id, "delete_file", {"path": "/tmp/a"})
    assert not ok and "expired" in reason.lower()


def test_all_actions_bind_parameters(custom_validator):
    req = custom_validator.request_approval("test_action", "temporary test", {"target": "one"})
    assert custom_validator.approve_action(req.id)
    ok, reason = custom_validator.consume_approval(req.id, "test_action", {"target": "two"})
    assert not ok and "parameter mismatch" in reason.lower()


def test_concurrent_approval_consumption_is_single_use(custom_validator):
    req = custom_validator.request_approval("test_action", "temporary test", {"target": "one"})
    assert custom_validator.approve_action(req.id)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: custom_validator.consume_approval(req.id, "test_action", {"target": "one"})[0], range(8)))
    assert results.count(True) == 1


def test_terminal_whitelisting_and_rejections():
    """Verify is_terminal_action_allowed accepts safe commands and strictly blocks dangerous commands."""
    # 1. Allowed commands
    assert ActionValidator.is_terminal_action_allowed("ls data/") is True
    assert ActionValidator.is_terminal_action_allowed("ls -la") is True
    assert ActionValidator.is_terminal_action_allowed("cat notes.txt") is True
    assert ActionValidator.is_terminal_action_allowed("echo 'Hello World'") is True
    assert ActionValidator.is_terminal_action_allowed("grep 'pattern' lecture.txt") is True
    assert ActionValidator.is_terminal_action_allowed("pytest") is True
    assert ActionValidator.is_terminal_action_allowed("ollama list") is True
    assert ActionValidator.is_terminal_action_allowed("npm run build") is True
    assert ActionValidator.is_terminal_action_allowed("python -c \"print(2 + 2)\"") is True

    # 2. Hard fail blocked commands
    assert ActionValidator.is_terminal_action_allowed("rm -rf /") is False
    assert ActionValidator.is_terminal_action_allowed("rm -rf data/") is False
    assert ActionValidator.is_terminal_action_allowed("cat /etc/passwd") is False
    assert ActionValidator.is_terminal_action_allowed("cat /var/log/syslog") is False
    assert ActionValidator.is_terminal_action_allowed("cat ~/.ssh/id_rsa") is False
    assert ActionValidator.is_terminal_action_allowed("sudo apt-get update") is False
    assert ActionValidator.is_terminal_action_allowed("chmod 777 script.sh") is False
    assert ActionValidator.is_terminal_action_allowed("chown user:user file") is False
    assert ActionValidator.is_terminal_action_allowed("dd if=/dev/zero of=/dev/sda") is False
    assert ActionValidator.is_terminal_action_allowed("mkfs.ext4 /dev/sda1") is False


def test_shell_injection_prevention():
    """Verify that shell injection characters and command chaining are strictly rejected."""
    # Semicolon chaining
    assert ActionValidator.is_terminal_action_allowed("echo hello; rm -rf /") is False
    assert ActionValidator.is_terminal_action_allowed("ls; cat /etc/passwd") is False

    # Logical operators (&&, ||)
    assert ActionValidator.is_terminal_action_allowed("echo hello && rm -rf /") is False
    assert ActionValidator.is_terminal_action_allowed("ls || rm -rf /") is False

    # Pipe (|)
    assert ActionValidator.is_terminal_action_allowed("cat data.txt | grep secret") is False

    # Redirections (>, <)
    assert ActionValidator.is_terminal_action_allowed("echo dangerous > /etc/hosts") is False
    assert ActionValidator.is_terminal_action_allowed("cat < /etc/shadow") is False

    # Subshell execution ($(), backticks)
    assert ActionValidator.is_terminal_action_allowed("echo $(whoami)") is False
    assert ActionValidator.is_terminal_action_allowed("echo `whoami`") is False


def test_execute_code_timeout():
    """Verify that execute_code enforces a hard timeout and returns a timeout error for runaway scripts."""
    infinite_loop = "while True:\n    pass"
    res = execute_code(infinite_loop, language="python", timeout=1.0, force_ast=True)
    assert res["status"] == "timeout"
    assert "timeout" in str(res["error"]).lower() or "timed out" in str(res["stderr"]).lower()


def test_execute_code_ast_fallback_rejects_forbidden_imports():
    """Verify that AST validation in execute_code strictly rejects dangerous module imports."""
    # 1. Reject import os
    res_os = execute_code("import os\nos.system('ls')", language="python", force_ast=True)
    assert res_os["status"] == "rejected"
    assert "os" in res_os["error"].lower() or "forbidden" in res_os["error"].lower()

    # 2. Reject import subprocess
    res_sub = execute_code("import subprocess\nsubprocess.run(['ls'])", language="python", force_ast=True)
    assert res_sub["status"] == "rejected"
    assert "subprocess" in res_sub["error"].lower() or "forbidden" in res_sub["error"].lower()

    # 3. Reject from-import shutil
    res_shutil = execute_code("from shutil import rmtree\nrmtree('/tmp')", language="python", force_ast=True)
    assert res_shutil["status"] == "rejected"
    assert "shutil" in res_shutil["error"].lower() or "forbidden" in res_shutil["error"].lower()

    # 4. Reject socket
    res_socket = execute_code("import socket\ns = socket.socket()", language="python", force_ast=True)
    assert res_socket["status"] == "rejected"
    assert "socket" in res_socket["error"].lower() or "forbidden" in res_socket["error"].lower()

    # 5. Reject forbidden builtin eval/exec
    res_eval = execute_code("eval('1 + 1')", language="python", force_ast=True)
    assert res_eval["status"] == "rejected"
    assert "eval" in res_eval["error"].lower() or "forbidden" in res_eval["error"].lower()


def test_execute_code_safe_execution_succeeds():
    """Verify that benign algorithmic and mathematical code executes cleanly."""
    safe_code = "print(sum(x * x for x in range(10)))"
    res = execute_code(safe_code, language="python", timeout=5.0, force_ast=True)
    assert res["status"] == "success"
    assert res["stdout"].strip() == "285"
    assert res["returncode"] == 0


def test_approval_required_for_side_effects():
    """Verify side-effect actions require approval while read-only actions do not."""
    # Side-effect actions require approval
    assert ActionValidator.requires_approval("file_write") is True
    assert ActionValidator.requires_approval("file_delete") is True
    assert ActionValidator.requires_approval("execute_code") is True
    assert ActionValidator.requires_approval("execute_shell_command") is True

    # Read-only actions do not require approval
    assert ActionValidator.requires_approval("file_read") is False
    assert ActionValidator.requires_approval("list_dir") is False

    # Approval token generation
    token = ActionValidator.generate_approval_token(
        action="file_delete",
        description="Delete temporary output",
        params={"path": "/workspace/temp.txt"}
    )
    assert token is not None
    assert len(token) > 10

    # Retrieve and verify token is in pending state
    req = ActionValidator.get_approval(token)
    assert req is not None
    assert req.action == "file_delete"
    assert req.status == ApprovalStatus.PENDING


def test_is_path_safe_rejects_traversal_and_allows_nested_workspace(temp_workspace):
    """Verify is_path_safe rejects ../, ../../, and absolute external paths, while allowing deeply nested paths."""
    validator = SecurityValidator(allowed_roots=[temp_workspace])

    # 1. Rejects relative traversal attempts escaping the workspace
    assert validator.is_path_safe("../secret.txt") is False
    assert validator.is_path_safe("../../etc/passwd") is False
    assert validator.is_path_safe(str(temp_workspace) + "/../../../etc/passwd") is False
    assert validator.is_path_safe(temp_workspace / ".." / "parent.txt") is False

    # 2. Rejects absolute paths outside the workspace
    assert validator.is_path_safe("/etc/passwd") is False
    assert validator.is_path_safe("/var/log/syslog") is False
    assert validator.is_path_safe(Path.home() / ".ssh" / "id_rsa") is False

    # 3. Accepts files directly in the workspace
    file_in_ws = temp_workspace / "assignment1.pdf"
    assert validator.is_path_safe(file_in_ws) is True
    assert validator.is_path_safe(str(file_in_ws)) is True

    # 4. Strictly accepts deeply nested subdirectories within the workspace
    deeply_nested_file = temp_workspace / "courses" / "cs101" / "week4" / "lectures" / "notes.txt"
    assert validator.is_path_safe(deeply_nested_file) is True
    assert validator.is_path_safe(str(deeply_nested_file)) is True

    # 5. Also verify via ActionValidator
    from app.security.validator import security_validator
    security_validator.allowed_roots.append(temp_workspace.resolve())
    assert ActionValidator.is_path_safe(deeply_nested_file) is True
    assert ActionValidator.is_path_safe("../../../etc/shadow") is False


def test_approval_token_parameter_binding():
    """Verify approval tokens are cryptographically/in-memory bound to specific parameters and rejected on alteration."""
    token = ActionValidator.generate_approval_token(
        action_name="file_delete",
        doc_id="123"
    )
    assert token is not None

    # Approve token
    assert ActionValidator.approve_action(token) is True

    # Attempt to consume token with altered parameters (e.g. doc_id="456") must be rejected
    res_mismatch = ActionValidator.consume_approval_token(
        token=token,
        action_name="file_delete",
        doc_id="456"
    )
    assert not res_mismatch
    assert res_mismatch == False
    assert "parameter mismatch" in str(res_mismatch.reason).lower()

    # Attempt to consume token with mismatched action must be rejected
    res_action_mismatch = ActionValidator.consume_approval_token(
        token=token,
        action_name="execute_code",
        doc_id="123"
    )
    assert not res_action_mismatch
    assert "mismatch" in str(res_action_mismatch.reason).lower()

    # Consuming with the exact bound parameters succeeds
    res_success = ActionValidator.consume_approval_token(
        token=token,
        action_name="file_delete",
        doc_id="123"
    )
    assert res_success
    assert res_success == True
    assert res_success.reason is None


def test_approval_token_single_use_replay_prevention():
    """Verify that once consumed, an approval token cannot be reused (replay prevention)."""
    token = ActionValidator.generate_approval_token(
        action_name="file_delete",
        doc_id="target_999"
    )
    assert ActionValidator.approve_action(token) is True

    # First consumption succeeds
    first_res = ActionValidator.consume_approval_token(token, "file_delete", doc_id="target_999")
    assert first_res == True

    # Second consumption with identical parameters must be rejected (replay prevention)
    replay_res = ActionValidator.consume_approval_token(token, "file_delete", doc_id="target_999")
    assert not replay_res
    assert replay_res == False
    assert "consumed" in str(replay_res.reason).lower() or "replay" in str(replay_res.reason).lower()


def test_approval_token_expiration():
    """Verify that an expired approval token cannot be consumed."""
    token = ActionValidator.generate_approval_token(
        action_name="file_delete",
        doc_id="expired_doc"
    )
    assert ActionValidator.approve_action(token) is True

    req = ActionValidator.get_approval(token)
    assert req is not None
    # Simulate timeout by adjusting requested_at in the past
    req.requested_at = time.time() - settings.ACTION_TIMEOUT_SECONDS - 10

    res = ActionValidator.consume_approval_token(token, "file_delete", doc_id="expired_doc")
    assert not res
    assert res == False
    assert "expired" in str(res.reason).lower()


def test_file_service_path_traversal_and_operations(temp_workspace, isolated_sandbox):
    """Verify FileService enforces is_path_safe, raises SecurityError on traversal, and handles safe file operations."""
    validator = SecurityValidator(allowed_roots=[temp_workspace])
    service = FileService(validator=validator, audit_logger_obj=isolated_sandbox.audit_logger)

    # 1. Path traversal read is blocked with SecurityError
    with pytest.raises(SecurityError) as exc_info:
        service.read_file("../../etc/passwd")
    assert "path traversal" in str(exc_info.value).lower() or "access denied" in str(exc_info.value).lower()

    # 2. Path traversal write is blocked with SecurityError
    with pytest.raises(SecurityError):
        service.write_file("../escape.txt", "malicious payload")

    # 3. Path traversal delete is blocked with SecurityError
    with pytest.raises(SecurityError):
        service.delete_file("../../../etc/shadow")

    # 4. Safe operations inside workspace
    nested_dir = temp_workspace / "nested" / "docs"
    safe_path = nested_dir / "lesson.txt"

    # Write file (generates approval if required or write directly if auto-approved)
    token = validator.generate_approval_token("file_write", params={"path": str(safe_path.resolve())}, auto_approve=True)
    write_res = service.write_file(safe_path, "Hello Academic World", approval_token=token)
    assert write_res["status"] == "success"
    assert safe_path.exists()

    # Read safe file
    read_content = service.read_file(safe_path)
    assert read_content == "Hello Academic World"

    # Delete safe file requires approval
    del_req = service.delete_file(safe_path)
    assert del_req["status"] == "approval_required"
    assert safe_path.exists()

    # Approve and delete
    del_token = del_req["approval_id"]
    validator.approve_action(del_token)
    del_res = service.delete_file(safe_path, approval_token=del_token)
    assert del_res["status"] == "success"
    assert not safe_path.exists()

