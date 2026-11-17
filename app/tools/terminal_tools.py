"""Restricted terminal execution tool with strict whitelisting, async execution, and approval gates."""

import asyncio
import concurrent.futures
from pathlib import Path
import shlex
from typing import Any, Dict, List, Optional

from app.security.validator import ActionRisk, ActionValidator, SecurityValidator, security_validator
from app.security.audit import audit_logger


class AsyncDictResult(dict):
    """Dictionary result that can also be awaited seamlessly in async contexts."""

    def __await__(self):
        async def _resolve():
            return self
        return _resolve().__await__()


class RestrictedTerminalTools:
    """Safe terminal execution preventing arbitrary command injection."""

    SAFE_COMMANDS = {"ls", "date", "pwd", "stat", "head", "tail", "wc", "echo", "tree", "grep", "cat", "pytest"}

    @classmethod
    def _get_validator(cls) -> SecurityValidator:
        from app.security.validator import security_validator as sec_val
        return sec_val

    @classmethod
    def _get_audit_logger(cls) -> Any:
        from app.security.audit import audit_logger as al
        return al

    @classmethod
    def _log_event(cls, **kwargs) -> None:
        try:
            logger_inst = cls._get_audit_logger()
            if logger_inst and hasattr(logger_inst, "log_event"):
                logger_inst.log_event(**kwargs)
        except Exception:
            pass

    @classmethod
    async def _run_subprocess_async(cls, command_str: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Execute subprocess using asyncio.create_subprocess_shell."""
        proc = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "returncode": proc.returncode,
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace")
            }
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise TimeoutError(f"Command timed out after {timeout}s")

    @classmethod
    def _execute_async_helper(cls, coro: Any) -> Dict[str, Any]:
        """Run an async coroutine synchronously, whether or not an event loop is currently active."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    @classmethod
    def run_safe_command(
        cls,
        command_str: str,
        approval_token: Optional[str] = None,
        approval_id: Optional[str] = None,
        timeout: float = 5.0,
        require_approval: bool = False
    ) -> AsyncDictResult:
        """Execute only explicitly whitelisted safe inspection commands using async subprocess."""
        sec_val = cls._get_validator()

        if not command_str or not isinstance(command_str, str) or not command_str.strip():
            return AsyncDictResult({"status": "error", "error": "Empty command"})

        cmd = command_str.strip()

        # 1. Check against security validator's strict whitelist and shell injection filter
        if not sec_val.is_terminal_action_allowed(cmd):
            cls._log_event(
                event_type="terminal",
                action="blocked_unapproved_command",
                details={"command": cmd},
                risk_level="high",
                status="blocked",
                error=f"Command '{cmd}' is not allowed by whitelist or contains unsafe patterns."
            )
            return AsyncDictResult({
                "status": "blocked",
                "error": f"Command not allowed: '{cmd}' violates security policy, contains forbidden patterns, or is not in whitelist."
            })

        # 2. Tokenize and verify base binary is safe
        try:
            tokens = shlex.split(cmd)
        except Exception as e:
            return AsyncDictResult({"status": "error", "error": f"Invalid command syntax: {e}"})

        if not tokens:
            return AsyncDictResult({"status": "error", "error": "Empty command"})

        binary = tokens[0].lower().split("/")[-1]
        if binary not in cls.SAFE_COMMANDS:
            cls._log_event(
                event_type="terminal",
                action="blocked_unapproved_command",
                details={"command": cmd, "binary": binary},
                risk_level="high",
                status="blocked",
                error=f"Binary '{binary}' is not in safe command whitelist"
            )
            return AsyncDictResult({
                "status": "blocked",
                "error": f"Command '{binary}' is not whitelisted. Safe commands: {sorted(list(cls.SAFE_COMMANDS))}"
            })

        # 3. Validate arguments do not target sensitive files or escape allowed roots
        if binary in ("ls", "cat", "head", "tail", "wc", "stat", "tree", "grep"):
            for arg in tokens[1:]:
                if arg.startswith("-"):
                    continue
                try:
                    candidate = Path(arg).resolve()
                    if sec_val.is_sensitive_target(candidate) or not sec_val.is_path_safe(candidate):
                        return AsyncDictResult({
                            "status": "blocked",
                            "error": f"Argument '{arg}' points outside authorized workspace directory or targets protected system/secret files."
                        })
                except Exception:
                    return AsyncDictResult({
                        "status": "blocked",
                        "error": f"Invalid path argument '{arg}'."
                    })
        else:
            for arg in tokens[1:]:
                if arg.startswith("-"):
                    continue
                try:
                    candidate = Path(arg).resolve()
                    if sec_val.is_sensitive_target(candidate):
                        return AsyncDictResult({
                            "status": "blocked",
                            "error": f"Argument '{arg}' targets protected system/secret files."
                        })
                except Exception:
                    pass

        # 4. Enforce approval token validation if token provided or required
        token = approval_token or approval_id
        if token:
            ok, err_msg = sec_val.consume_approval(token, "terminal_run", {"command": cmd})
            if not ok:
                ok, err_msg = sec_val.consume_approval(token, "execute_shell_command", {"command": cmd})
            if not ok:
                cls._log_event(
                    event_type="security_approval",
                    action="unapproved_command_attempt",
                    details={"command": cmd, "approval_id": token},
                    risk_level="high",
                    status="blocked",
                    error=err_msg
                )
                return AsyncDictResult({
                    "status": "blocked",
                    "error": f"Action approval rejected: {err_msg}"
                })
        elif require_approval:
            req = sec_val.request_approval(
                action="terminal_run",
                description=f"Request to run terminal command: '{cmd}'",
                params={"command": cmd},
                risk=ActionRisk.HIGH
            )
            return AsyncDictResult({
                "status": "approval_required",
                "approval_id": req.id,
                "message": f"Execution requires human approval. Approve request ID '{req.id}'."
            })

        # 5. Execute command using asyncio.create_subprocess_shell
        try:
            exec_res = cls._execute_async_helper(cls._run_subprocess_async(cmd, timeout=timeout))
            rc = exec_res["returncode"]
            cls._log_event(
                event_type="terminal",
                action="run_safe_command",
                details={"command": cmd, "returncode": rc},
                risk_level="low",
                status="success" if rc == 0 else "failed"
            )
            return AsyncDictResult({
                "status": "success",
                "returncode": rc,
                "stdout": exec_res["stdout"],
                "stderr": exec_res["stderr"]
            })
        except TimeoutError:
            return AsyncDictResult({"status": "error", "error": f"Command timed out after {timeout}s"})
        except Exception as e:
            return AsyncDictResult({"status": "error", "error": str(e)})

    @classmethod
    def execute_with_approval(cls, command_str: str, approval_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute arbitrary or modifying shell commands only with explicit approval token."""
        sec_val = cls._get_validator()
        if not approval_id:
            req = sec_val.request_approval(
                action="execute_shell_command",
                description=f"Execute potentially dangerous shell command: '{command_str}'",
                params={"command": command_str},
                risk=ActionRisk.HIGH
            )
            cls._log_event(
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

        ok, err_msg = sec_val.consume_approval(
            request_id=approval_id,
            expected_action="execute_shell_command",
            expected_params={"command": command_str}
        )
        if not ok:
            cls._log_event(
                event_type="security_approval",
                action="unapproved_command_attempt",
                details={"command": command_str, "approval_id": approval_id},
                risk_level="high",
                status="blocked",
                error=err_msg
            )
            raise PermissionError(f"Command execution rejected: {err_msg}")

        try:
            exec_res = cls._execute_async_helper(cls._run_subprocess_async(command_str, timeout=30.0))
            rc = exec_res["returncode"]
            cls._log_event(
                event_type="terminal",
                action="execute_approved_command",
                details={"command": command_str, "returncode": rc, "approval_id": approval_id},
                risk_level="high",
                status="success" if rc == 0 else "failed"
            )
            return {
                "status": "success",
                "returncode": rc,
                "stdout": exec_res["stdout"],
                "stderr": exec_res["stderr"]
            }
        except TimeoutError:
            return {"status": "error", "error": "Command timed out after 30s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @classmethod
    def get_tool_schemas(cls) -> List[Dict[str, Any]]:
        """Return OpenAI function calling JSON schema definition for terminal tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_safe_command",
                    "description": "Execute a whitelisted, non-destructive terminal command (e.g. ls, pwd, echo, cat, grep, date) inside the sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command string to execute."
                            },
                            "approval_token": {
                                "type": "string",
                                "description": "Optional human-in-the-loop approval token."
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Command execution timeout in seconds (default: 5.0)."
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]


TerminalTools = RestrictedTerminalTools
terminal_tools = RestrictedTerminalTools()
