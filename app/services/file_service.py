"""Secure file service with path traversal protection and approval workflow enforcement."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.security.validator import (
    ActionRisk,
    ActionValidator,
    ApprovalResult,
    ApprovalStatus,
    SecurityValidator,
    security_validator,
)

logger = logging.getLogger(__name__)


class SecurityError(PermissionError):
    """Exception raised when a file operation violates security constraints such as path traversal."""
    pass


class FileService:
    """Service providing secure file operations guarded by path traversal checks and approval tokens."""

    def __init__(
        self,
        validator: Optional[SecurityValidator] = None,
        audit_logger_obj: Optional[Any] = None
    ):
        self._validator = validator
        self._audit_logger = audit_logger_obj

    @property
    def validator(self) -> SecurityValidator:
        from app.security.validator import security_validator as sec_val
        return self._validator or sec_val

    @property
    def audit_logger(self) -> Any:
        if self._audit_logger is not None:
            return self._audit_logger
        from app.security import audit
        return getattr(audit, "audit_logger", None)

    def _get_validator(self) -> SecurityValidator:
        return self.validator

    def _log_event(self, **kwargs) -> None:
        try:
            logger_inst = self.audit_logger
            if logger_inst and hasattr(logger_inst, "log_event"):
                logger_inst.log_event(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")

    def is_path_safe(self, path: Union[str, Path]) -> bool:
        """Check if a path is safe from traversal attacks and within allowed roots."""
        return self._get_validator().is_path_safe(path)

    def read_file(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Safely read content from an authorized file path.

        Raises SecurityError if path traversal is detected or path is unsafe.
        Raises FileNotFoundError if file does not exist.
        """
        if not self.is_path_safe(path):
            self._log_event(
                event_type="security_violation",
                action="file_read_blocked",
                details={"path": str(path)},
                risk_level="high",
                status="blocked",
                error="Path traversal or unauthorized path access attempt"
            )
            raise SecurityError(f"Access denied: path traversal or unsafe path detected: '{path}'")

        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: '{resolved}'")
        if not resolved.is_file():
            raise ValueError(f"Path is not a regular file: '{resolved}'")

        content = resolved.read_text(encoding=encoding, errors="replace")
        self._log_event(
            event_type="filesystem",
            action="file_read",
            details={"path": str(resolved), "size": len(content)},
            risk_level="low",
            status="success"
        )
        return content

    def write_file(
        self,
        path: Union[str, Path],
        content: str,
        approval_token: Optional[str] = None,
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Safely write content to an authorized file path.

        Requires human-in-the-loop approval token if write side-effects require approval.
        Raises SecurityError if path traversal is detected or path is unsafe.
        """
        if not self.is_path_safe(path):
            self._log_event(
                event_type="security_violation",
                action="file_write_blocked",
                details={"path": str(path)},
                risk_level="high",
                status="blocked",
                error="Path traversal or unauthorized path access attempt"
            )
            raise SecurityError(f"Access denied: path traversal or unsafe path detected: '{path}'")

        resolved = Path(path).resolve()

        if self._get_validator().requires_approval("file_write"):
            if not approval_token:
                req = self._get_validator().request_approval(
                    action="file_write",
                    description=f"Request to write file '{resolved.name}' at {resolved}",
                    params={"path": str(resolved)},
                    risk=ActionRisk.MEDIUM
                )
                self._log_event(
                    event_type="security_approval",
                    action="request_file_write",
                    details={"path": str(resolved), "approval_id": req.id},
                    risk_level="medium",
                    status="pending_approval"
                )
                return {
                    "status": "approval_required",
                    "approval_id": req.id,
                    "message": f"Action blocked: Writing files requires explicit approval. Submit approval ID {req.id}."
                }

            ok, err_msg = self._get_validator().consume_approval(
                request_id=approval_token,
                expected_action="file_write",
                expected_params={"path": str(resolved)}
            )
            if not ok:
                self._log_event(
                    event_type="security_approval",
                    action="file_write_blocked",
                    details={"path": str(resolved), "approval_id": approval_token},
                    risk_level="medium",
                    status="blocked",
                    error=err_msg
                )
                raise SecurityError(f"Action not authorized: {err_msg}")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)

        self._log_event(
            event_type="filesystem",
            action="file_write",
            details={"path": str(resolved), "bytes_written": len(content)},
            risk_level="medium",
            status="success"
        )
        return {
            "status": "success",
            "path": str(resolved),
            "bytes_written": len(content),
            "message": f"File '{resolved}' successfully written."
        }

    def delete_file(
        self,
        path: Union[str, Path],
        approval_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Safely delete a file at an authorized file path.

        Strictly requires human-in-the-loop approval authorization.
        Raises SecurityError if path traversal is detected or path is unsafe.
        """
        if not self.is_path_safe(path):
            self._log_event(
                event_type="security_violation",
                action="file_delete_blocked",
                details={"path": str(path)},
                risk_level="high",
                status="blocked",
                error="Path traversal or unauthorized path access attempt"
            )
            raise SecurityError(f"Access denied: path traversal or unsafe path detected: '{path}'")

        resolved = Path(path).resolve()

        if not approval_token:
            req = self._get_validator().request_approval(
                action="delete_file",
                description=f"Request to delete file '{resolved.name}' at {resolved}",
                params={"path": str(resolved)},
                risk=ActionRisk.HIGH
            )
            self._log_event(
                event_type="security_approval",
                action="request_delete_file",
                details={"path": str(resolved), "approval_id": req.id},
                risk_level="high",
                status="pending_approval"
            )
            return {
                "status": "approval_required",
                "approval_id": req.id,
                "message": f"Action blocked: Deleting files requires explicit approval. Submit approval ID {req.id}."
            }

        ok, err_msg = self._get_validator().consume_approval(
            request_id=approval_token,
            expected_action="delete_file",
            expected_params={"path": str(resolved)}
        )
        if not ok:
            self._log_event(
                event_type="security_approval",
                action="delete_file_blocked",
                details={"path": str(resolved), "approval_id": approval_token},
                risk_level="high",
                status="blocked",
                error=err_msg
            )
            raise SecurityError(f"Action not authorized: {err_msg}")

        if resolved.exists():
            resolved.unlink()

        self._log_event(
            event_type="filesystem",
            action="file_delete",
            details={"path": str(resolved), "approval_id": approval_token},
            risk_level="high",
            status="success"
        )
        return {
            "status": "success",
            "path": str(resolved),
            "message": f"File '{resolved}' has been permanently removed."
        }


file_service = FileService()
