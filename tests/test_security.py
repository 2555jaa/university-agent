"""Unit tests for security validator, approval workflow, and audit logging."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from app.config import settings
from app.security.validator import SecurityValidator, ActionRisk, ApprovalStatus
from app.security.audit import AuditLogger
from app.tools.terminal_tools import RestrictedTerminalTools
from app.tools.filesystem_tools import FilesystemTools


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

