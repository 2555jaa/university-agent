"""Audit logging for tracking all agent activities, tool executions, and security decisions."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.config import settings


class AuditEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str  # e.g., "tool_execution", "approval_requested", "approval_granted", "rag_query"
    actor: str = "agent"  # "agent", "user", "system"
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"
    status: str = "success"  # "success", "failed", "blocked", "pending_approval"
    error: Optional[str] = None


class AuditLogger:
    """Manages append-only JSON line audit logs for forensic security and academic accountability."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.AUDIT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("university_agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        action: str,
        details: Dict[str, Any],
        actor: str = "agent",
        risk_level: str = "low",
        status: str = "success",
        error: Optional[str] = None
    ) -> AuditEntry:
        """Record an event to the audit log."""
        entry = AuditEntry(
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            risk_level=risk_level,
            status=status,
            error=error
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
        return entry

    def read_recent_events(self, limit: int = 50) -> list[dict]:
        """Read recent audit entries."""
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries[-limit:]
        except Exception as e:
            self.logger.error(f"Failed to read audit log: {e}")
            return []


audit_logger = AuditLogger()
