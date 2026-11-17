"""Security validator and approval workflow engine."""

import os
from pathlib import Path
import re
import shlex
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


class ApprovalResult(tuple):
    """Result of an approval consumption attempt, supporting tuple unpacking and truthiness."""

    def __new__(cls, success: bool, reason: Optional[str] = None):
        return super().__new__(cls, (bool(success), reason))

    @property
    def success(self) -> bool:
        return self[0]

    @property
    def reason(self) -> Optional[str]:
        return self[1]

    def __bool__(self) -> bool:
        return self[0]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self[0] == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"ApprovalResult(success={self[0]}, reason={self[1]!r})"


class SecurityValidator:
    """Validates paths, commands, and manages approvals for sensitive operations."""

    # Explicitly allowed safe CLI commands (read-only or inspection)
    SAFE_COMMAND_WHITELIST = {
        "ls", "cat", "head", "tail", "wc", "date", "echo", "pwd", "stat", "tree"
    }

    # Allowed academic/developer commands
    ALLOWED_TERMINAL_COMMANDS = {
        "python", "python3", "node", "ls", "cat", "echo", "grep",
        "ollama", "npm", "pytest", "pwd", "date", "head", "tail", "wc", "stat", "tree"
    }

    # Commands strictly requiring explicit approval / hard blocked
    DANGEROUS_COMMANDS = {
        "rm", "del", "dd", "mkfs", "chmod", "chown", "sudo", "su",
        "apt", "apt-get", "systemctl", "curl", "wget", "kill", "reboot", "shutdown"
    }

    HARD_BLOCKED_COMMANDS = {
        "sudo", "chmod", "chown", "mkfs", "dd", "su", "apt", "apt-get",
        "systemctl", "reboot", "shutdown", "kill", "rm"
    }

    # Regex patterns for blocked tokens and shell injections
    BLOCKED_TERMINAL_PATTERNS = [
        r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f",
        r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r",
        r"/etc(?:/|\b)",
        r"/var(?:/|\b)",
        r"~?/\.ssh(?:/|\b)",
        r"\.env(?:\b|$)",
        r";", r"&&", r"\|\|", r"\|", r">", r"<", r"`", r"\$\("
    ]

    SIDE_EFFECT_ACTIONS = {
        "file_write", "file_delete", "execute_code", "execute_shell_command",
        "terminal_run", "create_folder", "organize_materials", "delete_file"
    }

    READ_ONLY_ACTIONS = {
        "file_read", "list_dir", "read_file", "search_documents", "get_status"
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
        self._is_default_roots = (allowed_roots is None)
        if allowed_roots is not None:
            self._allowed_roots = [p.resolve() for p in allowed_roots]
        else:
            workspace_dir = getattr(self.settings, "WORKSPACE_DIR", None) or (self.settings.BASE_DIR / "workspace")
            workspace_dir = Path(workspace_dir).resolve()
            documents_dir = getattr(self.settings, "DOCUMENTS_DIR", None) or (
                (self.settings.DATA_DIR / "documents").resolve() if getattr(self.settings, "DATA_DIR", None) else None
            )
            roots = [workspace_dir]
            if documents_dir:
                roots.append(Path(documents_dir).resolve())
            self._allowed_roots = roots

        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        # Approval state is shared by API worker threads; consumption must be atomic.
        self._approval_lock = threading.RLock()

    @property
    def allowed_roots(self) -> List[Path]:
        return self._allowed_roots

    @allowed_roots.setter
    def allowed_roots(self, roots: List[Path]) -> None:
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
        app_dir = (self.settings.BASE_DIR / "app").resolve()
        if app_dir == resolved_target or app_dir in resolved_target.parents:
            return True
        tests_dir = (self.settings.BASE_DIR / "tests").resolve()
        if tests_dir == resolved_target or tests_dir in resolved_target.parents:
            return True
        scripts_dir = (self.settings.BASE_DIR / "scripts").resolve()
        if scripts_dir == resolved_target or scripts_dir in resolved_target.parents:
            return True
        return False

    def is_path_safe(self, requested_path: str | Path) -> bool:
        """Strictly verify requested path resolves within allowed directory roots and is not sensitive.

        Safely resolves the absolute path of requested_path using Path.resolve() and checks
        if it strictly is_relative_to() the allowed base directory (e.g., settings.DOCUMENTS_DIR
        or settings.WORKSPACE_DIR).
        If a path attempts to break out (e.g. ../../../etc/passwd or ../../home/user/keys),
        returns False. Deeply nested paths within allowed roots return True.
        """
        try:
            if not requested_path:
                return False
            resolved = Path(requested_path).resolve()

            # Reject sensitive application secrets, databases, and source files
            if self.is_sensitive_target(resolved):
                return False

            roots = list(self.allowed_roots)
            if getattr(self, "_is_default_roots", False):
                docs_dir = getattr(self.settings, "DOCUMENTS_DIR", None)
                if docs_dir and Path(docs_dir).resolve() not in roots:
                    roots.append(Path(docs_dir).resolve())
                ws_dir = getattr(self.settings, "WORKSPACE_DIR", None)
                if ws_dir and Path(ws_dir).resolve() not in roots:
                    roots.append(Path(ws_dir).resolve())

            for root in roots:
                root_resolved = Path(root).resolve()
                try:
                    if resolved.is_relative_to(root_resolved):
                        return True
                except AttributeError:
                    try:
                        resolved.relative_to(root_resolved)
                        return True
                    except ValueError:
                        pass
            return False
        except Exception:
            return False

    def is_safe_path(self, target_path: str | Path) -> bool:
        """Backwards-compatible alias for is_path_safe."""
        return self.is_path_safe(target_path)

    def validate_path_access(self, target_path: str | Path) -> Path:
        """Return resolved path if safe, otherwise raise PermissionError."""
        resolved = Path(target_path).resolve()
        if self.is_sensitive_target(resolved):
            raise PermissionError(f"Access denied: '{resolved}' is a sensitive system or configuration file.")
        if not self.is_path_safe(resolved):
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
        timeout = getattr(self.settings, "ACTION_TIMEOUT_SECONDS", 300)
        if time.time() - req.requested_at > timeout:
            req.status = ApprovalStatus.EXPIRED
            req.resolved_at = time.time()
            return False, "Approval token has expired"
        if req.status == ApprovalStatus.EXPIRED:
            return False, "Approval token has expired"
        if req.status == ApprovalStatus.CONSUMED or req.consumed:
            return False, "Approval token has already been consumed (replay prevention)"
        if req.status != ApprovalStatus.APPROVED:
            return False, f"Action not approved: current status is '{req.status.value}'"

        # Action binding
        if req.action != expected_action:
            return False, f"Action mismatch: approved for '{req.action}', attempted '{expected_action}'"

        # Parameter binding
        def canonical(params: Dict[str, Any]) -> Dict[str, str]:
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

    @classmethod
    def is_terminal_action_allowed(cls, command: str) -> bool:
        """Strictly validate whether a command is whitelisted, safe, and free from shell injections."""
        if not command or not isinstance(command, str):
            return False
        cmd = command.strip()
        if not cmd:
            return False

        # 1. Shell injection & blocked pattern checks
        for pat in cls.BLOCKED_TERMINAL_PATTERNS:
            if re.search(pat, cmd, re.IGNORECASE):
                return False

        # 2. Tokenize command
        try:
            tokens = shlex.split(cmd)
        except Exception:
            return False

        if not tokens:
            return False

        binary = tokens[0].lower().split("/")[-1]

        # Explicitly blocked command check
        if binary in cls.HARD_BLOCKED_COMMANDS:
            return False

        # Must be in allowed command whitelist
        if binary not in cls.ALLOWED_TERMINAL_COMMANDS:
            return False

        # Special check: npm must be 'npm run <script>'
        if binary == "npm":
            if len(tokens) < 2 or tokens[1].lower() != "run":
                return False

        # Check arguments for blocked path patterns
        for arg in tokens[1:]:
            for pat in [r"/etc(?:/|\b)", r"/var(?:/|\b)", r"~?/\.ssh(?:/|\b)", r"\.env"]:
                if re.search(pat, arg, re.IGNORECASE):
                    return False

        # Python command check: if -c snippet, check for dangerous imports
        if binary in ("python", "python3"):
            for i, arg in enumerate(tokens[1:]):
                if arg == "-c" and i + 1 < len(tokens[1:]):
                    snippet = tokens[1:][i + 1]
                    if re.search(r"\b(?:os|sys|subprocess|shutil|socket|pty|ctypes)\b", snippet):
                        return False

        return True

    def generate_approval_token(
        self,
        action: Optional[str] = None,
        description: Any = "",
        params: Optional[Dict[str, Any]] = None,
        risk: ActionRisk = ActionRisk.HIGH,
        *,
        action_name: Optional[str] = None,
        auto_approve: bool = False,
        **kwargs
    ) -> str:
        """Generate a single-use human-in-the-loop approval token for a sensitive action."""
        target_action = action or action_name
        if not target_action:
            raise ValueError("action or action_name must be provided")

        if isinstance(description, dict) and params is None:
            params = description
            description = ""

        merged_params: Dict[str, str] = {}
        if params and isinstance(params, dict):
            for k, v in params.items():
                merged_params[str(k)] = str(v)
        for k, v in kwargs.items():
            if k not in ("action", "action_name", "description", "risk", "auto_approve"):
                merged_params[str(k)] = str(v)

        req = self.request_approval(
            action=target_action,
            description=description or f"Approval required for {target_action}",
            params=merged_params,
            risk=risk
        )
        if auto_approve:
            req.status = ApprovalStatus.APPROVED
            req.resolved_at = time.time()
            req.reason = "Auto-approved during token generation"
        return req.id

    def consume_approval_token(
        self,
        token: str,
        action: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        *,
        action_name: Optional[str] = None,
        raise_on_error: bool = False,
        **kwargs
    ) -> ApprovalResult:
        """Validate and consume an approval token with strict parameter binding and single-use enforcement."""
        target_action = action or action_name
        if not target_action:
            raise ValueError("action or action_name must be provided")

        if isinstance(params, dict):
            merged_params = {str(k): str(v) for k, v in params.items()}
        else:
            merged_params = {}
        for k, v in kwargs.items():
            if k not in ("token", "action", "action_name", "params", "raise_on_error"):
                merged_params[str(k)] = str(v)

        ok, reason = self.consume_approval(
            request_id=token,
            expected_action=target_action,
            expected_params=merged_params
        )
        if not ok and raise_on_error:
            raise PermissionError(f"Approval rejected: {reason}")
        return ApprovalResult(ok, reason)

    def requires_approval(self, action: str) -> bool:
        """Check whether an action requires an explicit human-in-the-loop approval token."""
        if action in self.SIDE_EFFECT_ACTIONS:
            return True
        if action in self.READ_ONLY_ACTIONS:
            return False
        return True


security_validator = SecurityValidator()


class ActionValidator(SecurityValidator):
    """Action and security validator supporting class and instance method invocation."""

    @classmethod
    def is_terminal_action_allowed(cls, command: str) -> bool:
        return SecurityValidator.is_terminal_action_allowed(command)

    @classmethod
    def is_path_safe(cls, requested_path: str | Path) -> bool:
        from app.security.validator import security_validator as sec_val
        return sec_val.is_path_safe(requested_path)

    @classmethod
    def is_safe_path(cls, target_path: str | Path) -> bool:
        from app.security.validator import security_validator as sec_val
        return sec_val.is_path_safe(target_path)

    @classmethod
    def generate_approval_token(
        cls,
        action: Optional[str] = None,
        description: Any = "",
        params: Optional[Dict[str, Any]] = None,
        risk: ActionRisk = ActionRisk.HIGH,
        *,
        action_name: Optional[str] = None,
        auto_approve: bool = False,
        **kwargs
    ) -> str:
        from app.security.validator import security_validator as sec_val
        return sec_val.generate_approval_token(
            action=action,
            description=description,
            params=params,
            risk=risk,
            action_name=action_name,
            auto_approve=auto_approve,
            **kwargs
        )

    @classmethod
    def consume_approval_token(
        cls,
        token: str,
        action: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        *,
        action_name: Optional[str] = None,
        raise_on_error: bool = False,
        **kwargs
    ) -> ApprovalResult:
        from app.security.validator import security_validator as sec_val
        return sec_val.consume_approval_token(
            token=token,
            action=action,
            params=params,
            action_name=action_name,
            raise_on_error=raise_on_error,
            **kwargs
        )

    @classmethod
    def get_approval(cls, request_id: str) -> Optional[ApprovalRequest]:
        from app.security.validator import security_validator as sec_val
        return sec_val.get_approval(request_id)

    @classmethod
    def approve_action(cls, request_id: str, approver_note: Optional[str] = None) -> bool:
        from app.security.validator import security_validator as sec_val
        return sec_val.approve_action(request_id, approver_note)

    @classmethod
    def reject_action(cls, request_id: str, reason: Optional[str] = None) -> bool:
        from app.security.validator import security_validator as sec_val
        return sec_val.reject_action(request_id, reason)

    @classmethod
    def consume_approval(cls, request_id: str, expected_action: str, expected_params: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        from app.security.validator import security_validator as sec_val
        return sec_val.consume_approval(request_id, expected_action, expected_params)

    @classmethod
    def list_pending(cls) -> List[ApprovalRequest]:
        from app.security.validator import security_validator as sec_val
        return sec_val.list_pending()

    @classmethod
    def requires_approval(cls, action: str) -> bool:
        from app.security.validator import security_validator as sec_val
        return sec_val.requires_approval(action)


action_validator = security_validator


def is_path_safe(requested_path: str | Path) -> bool:
    return security_validator.is_path_safe(requested_path)


def is_safe_path(target_path: str | Path) -> bool:
    return security_validator.is_path_safe(target_path)


def generate_approval_token(*args, **kwargs) -> str:
    return security_validator.generate_approval_token(*args, **kwargs)


def consume_approval_token(*args, **kwargs) -> ApprovalResult:
    return security_validator.consume_approval_token(*args, **kwargs)
