"""Restricted filesystem tools with strict directory traversal prevention and approval gates."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.security.validator import security_validator, ActionRisk
from app.security.audit import audit_logger


class FilesystemTools:
    """Safe filesystem operations for academic material organization and reading."""

    @staticmethod
    def read_authorized_file(file_path: str) -> str:
        """Safely read content from an authorized academic file."""
        resolved = security_validator.validate_path_access(file_path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Path is not a regular file: {resolved}")

        content = resolved.read_text(encoding="utf-8", errors="replace")
        audit_logger.log_event(
            event_type="filesystem",
            action="read_file",
            details={"path": str(resolved), "size": len(content)},
            risk_level="low",
            status="success"
        )
        return content

    @staticmethod
    def create_academic_folder(folder_path: str, subject: Optional[str] = None) -> str:
        """Create a new folder inside the authorized workspace for organizing subject materials."""
        resolved = security_validator.validate_path_access(folder_path)
        resolved.mkdir(parents=True, exist_ok=True)

        audit_logger.log_event(
            event_type="filesystem",
            action="create_folder",
            details={"path": str(resolved), "subject": subject},
            risk_level="medium",
            status="success"
        )
        return f"Successfully created directory: {resolved}"

    @staticmethod
    def organize_materials(source_dir: str, target_dir: str) -> Dict[str, Any]:
        """Organize academic files by extension/type into designated subdirectories."""
        src = security_validator.validate_path_access(source_dir)
        dest = security_validator.validate_path_access(target_dir)

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

        audit_logger.log_event(
            event_type="filesystem",
            action="organize_materials",
            details={"source": str(src), "destination": str(dest), "count": len(moved_files)},
            risk_level="medium",
            status="success"
        )
        return {"total_organized": len(moved_files), "files": moved_files}

    @staticmethod
    def delete_file(file_path: str, approval_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a file. STRICTLY requires explicit approval authorization."""
        resolved = security_validator.validate_path_access(file_path)

        if not approval_id:
            req = security_validator.request_approval(
                action="delete_file",
                description=f"Request to delete file '{resolved.name}' at {resolved}",
                params={"path": str(resolved)},
                risk=ActionRisk.HIGH
            )
            audit_logger.log_event(
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

        ok, err_msg = security_validator.consume_approval(
            request_id=approval_id,
            expected_action="delete_file",
            expected_params={"path": str(resolved)}
        )
        if not ok:
            audit_logger.log_event(
                event_type="security_approval",
                action="delete_file_blocked",
                details={"path": str(resolved), "approval_id": approval_id},
                risk_level="high",
                status="blocked",
                error=err_msg
            )
            raise PermissionError(f"Action not authorized: {err_msg}")

        if resolved.exists():
            resolved.unlink()

        audit_logger.log_event(
            event_type="filesystem",
            action="delete_file",
            details={"path": str(resolved), "approval_id": approval_id},
            risk_level="high",
            status="success"
        )
        return {"status": "success", "message": f"File '{resolved}' has been permanently removed."}


filesystem_tools = FilesystemTools()
