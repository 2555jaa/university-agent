"""Restricted filesystem tools with strict directory traversal prevention and approval gates."""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from app.security.validator import ActionRisk, ActionValidator, SecurityValidator, security_validator
from app.services.file_service import FileService, SecurityError, file_service


class FilesystemTools:
    """Safe filesystem operations for academic material organization, reading, and writing."""

    @classmethod
    def _get_validator(cls) -> SecurityValidator:
        from app.security.validator import security_validator as sec_val
        return sec_val

    @classmethod
    def _get_file_service(cls) -> FileService:
        from app.services.file_service import file_service as fs
        return fs

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
    def read_file(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Safely read content from an authorized file path.

        Explicitly calls is_path_safe via FileService and returns a structured failure dict on SecurityError.
        """
        fs = cls._get_file_service()
        try:
            content = fs.read_file(path)
            resolved = Path(path).resolve()
            return {
                "status": "success",
                "path": str(resolved),
                "content": content
            }
        except SecurityError as e:
            return {
                "status": "blocked",
                "error": f"Path traversal detected: {e}"
            }
        except FileNotFoundError as e:
            return {
                "status": "error",
                "error": f"File not found: {e}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    @classmethod
    def write_file(
        cls,
        path: Union[str, Path],
        content: str,
        approval_token: Optional[str] = None,
        approval_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Safely write content to an authorized file path.

        Requires human-in-the-loop approval token and verifies path safety.
        """
        fs = cls._get_file_service()
        sec_val = cls._get_validator()
        token = approval_token or approval_id

        # Path traversal guard
        if not fs.is_path_safe(path):
            return {
                "status": "blocked",
                "error": f"Path traversal detected: '{path}' is outside allowed directories or sensitive."
            }

        resolved = Path(path).resolve()

        if not token:
            req = sec_val.request_approval(
                action="file_write",
                description=f"Request to write file '{resolved.name}' at {resolved}",
                params={"path": str(resolved)},
                risk=ActionRisk.MEDIUM
            )
            cls._log_event(
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

        try:
            return fs.write_file(path, content, approval_token=token)
        except SecurityError as e:
            if "path traversal" in str(e).lower() or not fs.is_path_safe(path):
                return {
                    "status": "blocked",
                    "error": f"Path traversal detected: {e}"
                }
            return {
                "status": "blocked",
                "error": f"Action not authorized: {e}"
            }
        except PermissionError as e:
            return {
                "status": "blocked",
                "error": f"Action not authorized: {e}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    @classmethod
    def list_directory(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """List files and subdirectories of an authorized directory within allowed roots."""
        fs = cls._get_file_service()
        if not fs.is_path_safe(path):
            return {
                "status": "blocked",
                "error": f"Path traversal detected: '{path}' is outside allowed directories or sensitive."
            }

        resolved = Path(path).resolve()
        if not resolved.exists():
            return {
                "status": "error",
                "error": f"Directory not found: '{resolved}'"
            }
        if not resolved.is_dir():
            return {
                "status": "error",
                "error": f"Path is not a directory: '{resolved}'"
            }

        entries = []
        files = []
        for p in sorted(resolved.iterdir(), key=lambda x: x.name):
            entries.append({
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": p.stat().st_size if p.is_file() else 0
            })
            files.append(p.name)

        return {
            "status": "success",
            "path": str(resolved),
            "entries": entries,
            "files": files
        }

    @classmethod
    def delete_file(
        cls,
        file_path: Union[str, Path],
        approval_token: Optional[str] = None,
        approval_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a file. STRICTLY requires explicit approval authorization."""
        fs = cls._get_file_service()
        sec_val = cls._get_validator()
        token = approval_token or approval_id

        if not fs.is_path_safe(file_path):
            return {
                "status": "blocked",
                "error": f"Path traversal detected: '{file_path}' is outside allowed directories or sensitive."
            }

        resolved = sec_val.validate_path_access(file_path)

        if not token:
            req = sec_val.request_approval(
                action="delete_file",
                description=f"Request to delete file '{resolved.name}' at {resolved}",
                params={"path": str(resolved)},
                risk=ActionRisk.HIGH
            )
            cls._log_event(
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

        ok, err_msg = sec_val.consume_approval(
            request_id=token,
            expected_action="delete_file",
            expected_params={"path": str(resolved)}
        )
        if not ok:
            ok, err_msg = sec_val.consume_approval(
                request_id=token,
                expected_action="file_delete",
                expected_params={"path": str(resolved)}
            )

        if not ok:
            cls._log_event(
                event_type="security_approval",
                action="delete_file_blocked",
                details={"path": str(resolved), "approval_id": token},
                risk_level="high",
                status="blocked",
                error=err_msg
            )
            raise PermissionError(f"Action not authorized: {err_msg}")

        if resolved.exists():
            resolved.unlink()

        cls._log_event(
            event_type="filesystem",
            action="delete_file",
            details={"path": str(resolved), "approval_id": token},
            risk_level="high",
            status="success"
        )
        return {"status": "success", "message": f"File '{resolved}' has been permanently removed."}

    # Backward-compatible helper methods
    @classmethod
    def read_authorized_file(cls, file_path: str) -> str:
        """Safely read content from an authorized academic file."""
        fs = cls._get_file_service()
        return fs.read_file(file_path)

    @classmethod
    def create_academic_folder(cls, folder_path: str, subject: Optional[str] = None) -> str:
        """Create a new folder inside the authorized workspace for organizing subject materials."""
        sec_val = cls._get_validator()
        resolved = sec_val.validate_path_access(folder_path)
        resolved.mkdir(parents=True, exist_ok=True)
        cls._log_event(
            event_type="filesystem",
            action="create_folder",
            details={"path": str(resolved), "subject": subject},
            risk_level="medium",
            status="success"
        )
        return f"Successfully created directory: {resolved}"

    @classmethod
    def organize_materials(cls, source_dir: str, target_dir: str) -> Dict[str, Any]:
        """Organize academic files by extension/type into designated subdirectories."""
        sec_val = cls._get_validator()
        src = sec_val.validate_path_access(source_dir)
        dest = sec_val.validate_path_access(target_dir)

        if not src.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {src}")

        dest.mkdir(parents=True, exist_ok=True)
        type_dirs = {
            "pdf": dest / "PDFs",
            "docx": dest / "Documents",
            "txt": dest / "Notes",
            "code": dest / "Code"
        }
        for d in type_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        moved_files = []
        for item in src.iterdir():
            if item.is_file():
                ext = item.suffix.lower()
                target_subdir = None
                if ext == ".pdf":
                    target_subdir = type_dirs["pdf"]
                elif ext in [".docx", ".doc"]:
                    target_subdir = type_dirs["docx"]
                elif ext in [".txt", ".md"]:
                    target_subdir = type_dirs["txt"]
                elif ext in [".py", ".c", ".cpp", ".java", ".js", ".html"]:
                    target_subdir = type_dirs["code"]

                if target_subdir:
                    new_path = target_subdir / item.name
                    shutil.copy2(item, new_path)
                    moved_files.append({"source": str(item), "destination": str(new_path)})

        cls._log_event(
            event_type="filesystem",
            action="organize_materials",
            details={"source": str(src), "destination": str(dest), "count": len(moved_files)},
            risk_level="medium",
            status="success"
        )
        return {"total_organized": len(moved_files), "files": moved_files}

    @classmethod
    def get_tool_schemas(cls) -> List[Dict[str, Any]]:
        """Return OpenAI function calling JSON schema definitions for filesystem tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from an authorized file inside the workspace or documents directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read."
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write text content to a file inside the authorized workspace directory. Requires approval token.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Target file path inside workspace."
                            },
                            "content": {
                                "type": "string",
                                "description": "Text content to write."
                            },
                            "approval_token": {
                                "type": "string",
                                "description": "Approval token authorized for file write."
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and subdirectories of an authorized workspace or documents directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to inspect."
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file inside the workspace. Strictly requires human approval token.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to delete."
                            },
                            "approval_token": {
                                "type": "string",
                                "description": "Single-use approval token authorized for file deletion."
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]


filesystem_tools = FilesystemTools()
