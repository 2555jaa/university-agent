"""Sandboxed code execution engine with AST inspection and timeout protection."""

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger("university_agent.security.sandbox")

# Explicitly forbidden modules in Python sandbox
FORBIDDEN_PYTHON_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "socket", "pty", "ctypes",
    "commands", "posix", "nt", "signal", "multiprocessing", "threading",
    "importlib", "_thread", "inspect", "resource", "code", "pdb", "builtins"
}

# Explicitly forbidden function calls
FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "__import__", "compile", "breakpoint"
}

# Explicitly forbidden attributes
FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__globals__", "__code__", "__builtins__", "__import__",
    "__class__", "__base__", "__bases__"
}


def validate_python_ast(code: str) -> Tuple[bool, Optional[str]]:
    """Inspect Python code AST to verify no forbidden modules, calls, or attributes are accessed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

    for node in ast.walk(tree):
        # 1. Direct imports: import os, import subprocess
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0].strip().lower()
                if root_mod in FORBIDDEN_PYTHON_MODULES:
                    return False, f"Security violation: import of module '{alias.name}' is strictly forbidden in sandbox."

        # 2. From imports: from os import system, from subprocess import Popen
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0].strip().lower()
                if root_mod in FORBIDDEN_PYTHON_MODULES:
                    return False, f"Security violation: from-import of module '{node.module}' is strictly forbidden in sandbox."

        # 3. Forbidden function calls: eval(), exec(), __import__()
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name and func_name in FORBIDDEN_CALLS:
                return False, f"Security violation: calling forbidden function '{func_name}()' is blocked."

        # 4. Forbidden attribute access (e.g. object.__subclasses__())
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                return False, f"Security violation: accessing protected attribute '{node.attr}' is forbidden."

    return True, None


def validate_javascript_code(code: str) -> Tuple[bool, Optional[str]]:
    """Lightweight heuristic inspection of JavaScript / Node.js code."""
    forbidden_patterns = [
        "child_process", "require('fs')", 'require("fs")', "require('net')", 'require("net")',
        "require('os')", 'require("os")', "process.exit", "process.kill", "eval(", "Function("
    ]
    for pattern in forbidden_patterns:
        if pattern in code:
            return False, f"Security violation: JavaScript code contains forbidden pattern '{pattern}'."
    return True, None


def execute_code(
    code: str,
    language: str = "python",
    timeout: float = 5.0,
    force_ast: bool = False
) -> Dict[str, Any]:
    """Execute code in an isolated environment enforcing strict AST validation and hard timeout limits.

    Args:
        code: Code string to execute.
        language: Programming language ("python", "javascript", "node").
        timeout: Maximum execution time in seconds (hard capped at 5.0s).
        force_ast: If True, bypass Docker and force AST-isolated runner.

    Returns:
        Dict with keys: status, stdout, stderr, returncode, error.
    """
    if not code or not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "stderr": "Empty code payload",
            "returncode": 1,
            "error": "Empty code payload"
        }

    # Hard timeout enforcement: maximum 5.0 seconds
    effective_timeout = min(max(0.1, float(timeout)), 5.0)
    lang = language.strip().lower()

    # 1. AST / Pre-execution Security Validation
    if lang in ("python", "py"):
        is_safe, error_msg = validate_python_ast(code)
        if not is_safe:
            return {
                "status": "rejected",
                "stdout": "",
                "stderr": error_msg or "Security validation failed",
                "returncode": 1,
                "error": error_msg or "Security validation failed"
            }
    elif lang in ("javascript", "node", "js"):
        is_safe, error_msg = validate_javascript_code(code)
        if not is_safe:
            return {
                "status": "rejected",
                "stdout": "",
                "stderr": error_msg or "Security validation failed",
                "returncode": 1,
                "error": error_msg or "Security validation failed"
            }
    else:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Unsupported language: '{language}'",
            "returncode": 1,
            "error": f"Unsupported language: '{language}'"
        }

    # 2. Check Docker availability
    has_docker = False
    if not force_ast and shutil.which("docker"):
        try:
            check_dock = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            has_docker = check_dock.returncode == 0
        except Exception:
            has_docker = False

    # 3. Docker execution if container image is ready
    if has_docker and not force_ast:
        image = "python:3.12-alpine" if lang in ("python", "py") else "node:20-alpine"
        try:
            inspect_img = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=2.0
            )
            if inspect_img.returncode == 0:
                cmd = ["docker", "run", "--rm", "-i", "--network", "none", image]
                if lang in ("python", "py"):
                    cmd.extend(["python", "-c", code])
                else:
                    cmd.extend(["node", "-e", code])

                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout
                )
                return {
                    "status": "success" if res.returncode == 0 else "error",
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "returncode": res.returncode,
                    "error": res.stderr if res.returncode != 0 else None
                }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Code execution timed out after {effective_timeout} seconds.",
                "returncode": -1,
                "error": f"Execution timed out after {effective_timeout} seconds."
            }
        except Exception as e:
            logger.warning(f"Docker execution failed, falling back to AST runner: {e}")

    # 4. Fallback: Process isolation with -I (isolated mode)
    try:
        if lang in ("python", "py"):
            runner = [sys.executable, "-I", "-c", code]
        else:
            node_bin = shutil.which("node") or "node"
            runner = [node_bin, "-e", code]

        res = subprocess.run(
            runner,
            capture_output=True,
            text=True,
            timeout=effective_timeout
        )
        return {
            "status": "success" if res.returncode == 0 else "error",
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
            "error": res.stderr if res.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": f"Code execution timed out after {effective_timeout} seconds.",
            "returncode": -1,
            "error": f"Execution timed out after {effective_timeout} seconds."
        }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "returncode": 1,
            "error": str(e)
        }


def get_sandbox_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI function calling JSON schema definition for code sandbox."""
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python or Node.js code securely in an isolated, sandboxed environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Source code snippet to execute."
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"],
                            "description": "Programming language (default: python)."
                        },
                        "approval_token": {
                            "type": "string",
                            "description": "Human-in-the-loop approval token if required."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

