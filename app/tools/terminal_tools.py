"""Restricted terminal execution tool with strict whitelisting and human approval gates."""

import subprocess
import shlex
from typing import Any, Dict, List, Optional

from app.security.validator import security_validator, ActionRisk
from app.security.audit import audit_logger


class RestrictedTerminalTools:
    """Safe terminal execution preventing arbitrary command injection."""

    SAFE_COMMANDS = {"ls", "date", "pwd", "stat", "head", "tail", "wc", "echo", "tree"}

    @classmethod
    def run_safe_command(cls, command_str: str) -> Dict[str, Any]:
        """Execute only explicitly whitelisted safe inspection commands without a shell."""
        try:
            tokens = shlex.split(command_str.strip())
        except Exception as e:
            return {"status": "error", "error": f"Invalid command syntax: {e}"}

        if not tokens:
            return {"status": "error", "error": "Empty command"}

        binary = tokens[0]
        if binary not in cls.SAFE_COMMANDS:
            audit_logger.log_event(
                event_type="terminal",
                action="blocked_unapproved_command",
                details={"command": command_str, "binary": binary},
                risk_level="high",
                status="blocked",
                error=f"Binary '{binary}' is not in safe command whitelist"
            )
            return {
                "status": "blocked",
                "error": f"Command '{binary}' is not whitelisted. Safe commands: {sorted(list(cls.SAFE_COMMANDS))}"
            }

        # Validate that arguments do not target sensitive files or escape allowed boundaries
        for arg in tokens[1:]:
            if arg.startswith("-"):
                continue
            try:
                candidate = Path(arg).resolve()
                if security_validator.is_sensitive_target(candidate) or not security_validator.is_safe_path(candidate):
                    return {
                        "status": "blocked",
                        "error": f"Argument '{arg}' points outside authorized workspace directory or targets protected system/secret files."
                    }
            except Exception:
                return {
                    "status": "blocked",
                    "error": f"Invalid path argument '{arg}'."
                }

        try:
            res = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False
            )
            audit_logger.log_event(
                event_type="terminal",
                action="run_safe_command",
                details={"command": command_str, "returncode": res.returncode},
                risk_level="low",
                status="success" if res.returncode == 0 else "failed"
            )
            return {
                "status": "success",
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Command timed out after 10s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @classmethod
    def execute_with_approval(cls, command_str: str, approval_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute arbitrary or modifying shell commands only with explicit approval token."""
        if not approval_id:
            req = security_validator.request_approval(
                action="execute_shell_command",
                description=f"Execute potentially dangerous shell command: '{command_str}'",
                params={"command": command_str},
                risk=ActionRisk.HIGH
            )
            audit_logger.log_event(
                event_type="security_approval",
                action="request_command_execution",
                details={"command": command_str, "approval_id": req.id},
                risk_level="high",
                status="pending_approval"
            )
            return {
                "status": "approval_required",
                "approval_id": req.id,
                "message": f"Execution requires human approval. Approve request ID '{req.id}'."
            }

        ok, err_msg = security_validator.consume_approval(
            request_id=approval_id,
            expected_action="execute_shell_command",
            expected_params={"command": command_str}
        )
        if not ok:
            audit_logger.log_event(
                event_type="security_approval",
                action="unapproved_command_attempt",
                details={"command": command_str, "approval_id": approval_id},
                risk_level="high",
                status="blocked",
                error=err_msg
            )
            raise PermissionError(f"Command execution rejected: {err_msg}")

        try:
            res = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            audit_logger.log_event(
                event_type="terminal",
                action="execute_approved_command",
                details={"command": command_str, "returncode": res.returncode, "approval_id": approval_id},
                risk_level="high",
                status="success" if res.returncode == 0 else "failed"
            )
            return {
                "status": "success",
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Command timed out after 30s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


terminal_tools = RestrictedTerminalTools()
