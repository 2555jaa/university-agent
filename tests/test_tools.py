"""Unit and integration tests for secure Terminal and Filesystem Tools."""

from pathlib import Path
import pytest

from app.security.validator import ActionValidator, security_validator
from app.tools.terminal_tools import TerminalTools, RestrictedTerminalTools
from app.tools.filesystem_tools import FilesystemTools


@pytest.fixture
def temp_workspace(tmp_path, isolated_sandbox):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    from app.security.validator import security_validator
    from app.services.file_service import file_service
    for val in (isolated_sandbox.security_validator, security_validator, file_service.validator):
        if val and ws.resolve() not in val.allowed_roots:
            val.allowed_roots.append(ws.resolve())
    return ws


def test_terminal_tools_run_safe_command_with_token():
    """Verify TerminalTools executes a safe whitelisted command with an approval token."""
    token = ActionValidator.generate_approval_token(
        action_name="terminal_run",
        command='echo "hello"',
        auto_approve=True
    )
    res = TerminalTools.run_safe_command('echo "hello"', approval_token=token)
    assert res["status"] == "success"
    assert "hello" in res["stdout"].strip()
    assert res["returncode"] == 0


def test_terminal_tools_blocks_malicious_command():
    """Verify TerminalTools strictly blocks non-whitelisted/dangerous commands and returns a structured error."""
    # 1. Block destructive rm command
    res_rm = TerminalTools.run_safe_command("rm -rf /")
    assert res_rm["status"] == "blocked"
    assert "error" in res_rm
    assert any(term in res_rm["error"].lower() for term in ["whitelist", "not allowed", "forbidden", "policy"])

    # 2. Block system commands (e.g. sudo)
    res_sudo = TerminalTools.run_safe_command("sudo ls")
    assert res_sudo["status"] == "blocked"

    # 3. Block command injection attempts
    res_inject = TerminalTools.run_safe_command("echo hello; rm -rf /")
    assert res_inject["status"] == "blocked"


def test_filesystem_tools_write_file_requires_token_and_succeeds(temp_workspace):
    """Verify FilesystemTools.write_file requires a valid approval token and creates the target file."""
    test_file = temp_workspace / "assignment_solution.txt"
    content = "print('Academic Agent')"

    # 1. Attempt without token -> approval_required
    req_res = FilesystemTools.write_file(str(test_file), content)
    assert req_res["status"] == "approval_required"
    assert "approval_id" in req_res
    token = req_res["approval_id"]

    # 2. Approve token
    assert ActionValidator.approve_action(token) is True

    # 3. Call write_file with valid token -> success
    write_res = FilesystemTools.write_file(str(test_file), content, approval_token=token)
    assert write_res["status"] == "success"
    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == content


def test_filesystem_tools_read_file_blocks_path_traversal():
    """Verify FilesystemTools.read_file catches SecurityError and returns a structured blocked error dict."""
    res = FilesystemTools.read_file("../../../etc/passwd")
    assert res["status"] == "blocked"
    assert "path traversal detected" in res["error"].lower() or "traversal" in res["error"].lower()


def test_filesystem_tools_read_file_and_list_directory(temp_workspace):
    """Verify safe read_file and list_directory operations inside workspace, and traversal rejection."""
    safe_file = temp_workspace / "notes.txt"
    safe_file.write_text("Calculus III Notes")

    # Read safe file
    read_res = FilesystemTools.read_file(str(safe_file))
    assert read_res["status"] == "success"
    assert read_res["content"] == "Calculus III Notes"

    # List directory
    list_res = FilesystemTools.list_directory(str(temp_workspace))
    assert list_res["status"] == "success"
    assert "notes.txt" in list_res["files"]
    assert any(e["name"] == "notes.txt" for e in list_res["entries"])

    # Path traversal on list directory blocked
    traversal_list = FilesystemTools.list_directory("../../etc")
    assert traversal_list["status"] == "blocked"


def test_filesystem_tools_delete_file_workflow(temp_workspace):
    """Verify FilesystemTools.delete_file requires approval, deletes file, and blocks path traversal."""
    target = temp_workspace / "to_delete.txt"
    target.write_text("cleanup")

    # 1. Attempt delete without token -> approval_required
    del_req = FilesystemTools.delete_file(str(target))
    assert del_req["status"] == "approval_required"
    token = del_req["approval_id"]

    # 2. Approve and delete
    assert ActionValidator.approve_action(token) is True
    del_res = FilesystemTools.delete_file(str(target), approval_token=token)
    assert del_res["status"] == "success"
    assert not target.exists()

    # 3. Path traversal on delete is blocked
    traversal_del = FilesystemTools.delete_file("../../../etc/shadow")
    assert traversal_del["status"] == "blocked"


def test_tool_schemas_for_llms():
    """Verify tool schema definitions follow OpenAI function calling format."""
    # Terminal tool schemas
    terminal_schemas = TerminalTools.get_tool_schemas()
    assert isinstance(terminal_schemas, list)
    assert len(terminal_schemas) >= 1
    assert terminal_schemas[0]["type"] == "function"
    assert terminal_schemas[0]["function"]["name"] == "run_safe_command"
    assert "parameters" in terminal_schemas[0]["function"]

    # Filesystem tool schemas
    fs_schemas = FilesystemTools.get_tool_schemas()
    assert isinstance(fs_schemas, list)
    assert len(fs_schemas) >= 4
    names = [s["function"]["name"] for s in fs_schemas]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names
    assert "delete_file" in names
    for schema in fs_schemas:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]
