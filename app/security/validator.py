"""Security validator and approval workflow engine."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import time
import threading
from enum import Enum
from pydantic import BaseModel, Field

from app.config import settings


class ActionRisk(str, Enum):
    LOW = "low"          # Safe, read-only or scoped operations
    MEDIUM = "medium"    # Modifying allowed local workspace
    HIGH = "high"        # File deletion, terminal execution, system modifications


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str
    description: str
    params: Dict[str, str] = Field(default_factory=dict)
    risk_level: ActionRisk = ActionRisk.HIGH
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: float = Field(default_factory=time.time)
    resolved_at: Optional[float] = None
    consumed_at: Optional[float] = None
    reason: Optional[str] = None
    consumed: bool = False


class SecurityValidator:
    """Validates paths, commands, and manages approvals for sensitive operations."""

    # Explicitly allowed safe CLI commands (read-only or inspection)
    SAFE_COMMAND_WHITELIST = {
        "ls", "cat", "head", "tail", "wc", "date", "echo", "pwd", "stat", "tree"
    }

    # Commands strictly requiring explicit approval
    DANGEROUS_COMMANDS = {
        "rm", "del", "dd", "mkfs", "chmod", "chown", "sudo", "su",
        "apt", "apt-get", "systemctl", "curl", "wget", "kill", "reboot", "shutdown"
    }

    # Sensitive filenames, extensions, and directory names strictly forbidden from normal access
    SENSITIVE_NAMES = {
        ".env", ".git", ".venv", ".pytest_cache", "__pycache__",
        "university_agent.db", "audit.log", "config.py"
    }
    SENSITIVE_EXTENSIONS = {
        ".py", ".pyc", ".sh", ".bash", ".key", ".pem", ".db", ".log"
    }

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        if allowed_roots is not None:
            self._custom_roots = True
            self._allowed_roots = [p.resolve() for p in allowed_roots]
        else:
            self._custom_roots = False
            self._allowed_roots = []

        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        # Approval state is shared by API worker threads; consumption must be atomic.
        self._approval_lock = threading.RLock()

    @property
    def allowed_roots(self) -> List[Path]:
        if self._custom_roots:
            return self._allowed_roots
        workspace_dir = (self.settings.BASE_DIR / "workspace").resolve()
        documents_dir = self.settings.DOCUMENTS_DIR.resolve()
        return [documents_dir, workspace_dir]

    @allowed_roots.setter
    def allowed_roots(self, roots: List[Path]) -> None:
        self._custom_roots = True
        self._allowed_roots = [p.resolve() for p in roots]

    def is_sensitive_target(self, resolved_target: Path) -> bool:
        """Check if a path targets sensitive application configuration, secrets, or source code."""
        # Check target name and parts
        for part in resolved_target.parts:
            if part in self.SENSITIVE_NAMES:
                return True
            if part.startswith(".env"):
                return True
        if resolved_target.name in self.SENSITIVE_NAMES:
            return True
        if resolved_target.suffix.lower() in self.SENSITIVE_EXTENSIONS:
            return True
        # Block access to app, tests, scripts dirs
        app_dir = (settings.BASE_DIR / "app").resolve()
        if app_dir == resolved_target or app_dir in resolved_target.parents:
            return True
        tests_dir = (settings.BASE_DIR / "tests").resolve()
        if tests_dir == resolved_target or tests_dir in resolved_target.parents:
            return True
        scripts_dir = (settings.BASE_DIR / "scripts").resolve()
        if scripts_dir == resolved_target or scripts_dir in resolved_target.parents:
            return True
        return False

    def is_safe_path(self, target_path: str | Path) -> bool:
        """Verify target path is resolved, contained within allowed directory roots, and not sensitive."""
        try:
            resolved_target = Path(target_path).resolve()
            if self.is_sensitive_target(resolved_target):
                return False
            for root in self.allowed_roots:
                if resolved_target == root or root in resolved_target.parents:
                    return True
            return False
        except Exception:
            return False

    def validate_path_access(self, target_path: str | Path) -> Path:
        """Return resolved path if safe, otherwise raise PermissionError."""
        resolved = Path(target_path).resolve()
        if self.is_sensitive_target(resolved):
            raise PermissionError(f"Access denied: '{resolved}' is a sensitive system or configuration file.")
        if not self.is_safe_path(resolved):
            raise PermissionError(
                f"Access denied: '{resolved}' is outside allowed directories: "
                f"{[str(r) for r in self.allowed_roots]}"
            )
        return resolved

    def request_approval(self, action: str, description: str, params: Dict[str, str], risk: ActionRisk = ActionRisk.HIGH) -> ApprovalRequest:
        """Create a new approval request for human-in-the-loop confirmation."""
        req = ApprovalRequest(
            action=action,
            description=description,
            params=params,
            risk_level=risk
        )
        with self._approval_lock:
            self._pending_approvals[req.id] = req
        return req

    def get_approval(self, request_id: str) -> Optional[ApprovalRequest]:
        """Fetch approval request by ID, checking expiration."""
        with self._approval_lock:
            req = self._pending_approvals.get(request_id)
            if req and req.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED):
                if time.time() - req.requested_at > settings.ACTION_TIMEOUT_SECONDS:
                    req.status = ApprovalStatus.EXPIRED
                    req.resolved_at = time.time()
            return req

    def approve_action(self, request_id: str, approver_note: Optional[str] = None) -> bool:
        """Approve a pending action request."""
        with self._approval_lock:
            req = self.get_approval(request_id)
            if not req or req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.APPROVED
            req.resolved_at = time.time()
            req.reason = approver_note or "User approved via API/UI"
            return True

    def reject_action(self, request_id: str, reason: Optional[str] = None) -> bool:
        """Reject a pending action request."""
        with self._approval_lock:
            req = self.get_approval(request_id)
            if not req or req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.REJECTED
            req.resolved_at = time.time()
            req.reason = reason or "User rejected action"
            return True

    def consume_approval(self, request_id: str, expected_action: str, expected_params: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """Validate approval authorization, enforce parameter/action binding, and consume token (single-use)."""
        with self._approval_lock:
            return self._consume_approval_locked(request_id, expected_action, expected_params)

    def _consume_approval_locked(self, request_id: str, expected_action: str, expected_params: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        req = self.get_approval(request_id)
        if not req:
            return False, "Approval token not found"
        if req.status == ApprovalStatus.CONSUMED or req.consumed:
            return False, "Approval token has already been consumed (replay prevention)"
        if req.status != ApprovalStatus.APPROVED:
            return False, f"Action not approved: current status is '{req.status.value}'"

        # Action binding
        if req.action != expected_action:
            return False, f"Action mismatch: approved for '{req.action}', attempted '{expected_action}'"

        # Parameter binding
        def canonical(params: Dict[str, str]) -> Dict[str, str]:
            result = {}
            for key, value in params.items():
                value = str(value)
                if key.lower() in {"path", "filepath", "file_path"}:
                    value = str(Path(value).resolve())
                elif key.lower() == "command":
                    value = value.strip()
                result[key] = value
            return result

        if canonical(req.params) != canonical(expected_params):
            return False, "Parameter mismatch: approved parameters do not match execution target"

        # Consume token immediately
        req.status = ApprovalStatus.CONSUMED
        req.consumed = True
        req.consumed_at = time.time()
        return True, None

    def list_pending(self) -> List[ApprovalRequest]:
        """Return list of all currently pending approval requests."""
        now = time.time()
        active = []
        with self._approval_lock:
            for req in self._pending_approvals.values():
                if req.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED) and now - req.requested_at > settings.ACTION_TIMEOUT_SECONDS:
                    req.status = ApprovalStatus.EXPIRED
                    req.resolved_at = now
                if req.status == ApprovalStatus.PENDING:
                    active.append(req)
        return active


security_validator = SecurityValidator()
