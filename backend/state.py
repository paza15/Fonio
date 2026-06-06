"""In-memory orchestrator state (single-process).

Holds the current recovery's working set so the dashboard can render it
without re-querying everything. The DB is still source of truth — this is
view-state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CurrentRecovery:
    slot_id: int
    phase: str
    started_at: datetime
    current_patient_id: Optional[int] = None
    current_patient_name: Optional[str] = None
    current_started_at: Optional[datetime] = None
    candidates: list = field(default_factory=list)  # list[Candidate dict]
    skipped: list = field(default_factory=list)
    tried_patient_ids: list[int] = field(default_factory=list)


class _State:
    def __init__(self):
        self.lock = threading.RLock()
        self.recovery: Optional[CurrentRecovery] = None
        self.pending_calls: dict[str, dict] = {}  # fonio_call_id → {slot_id,patient_id}
        self.webhook_events: dict[str, dict] = {}  # fonio_call_id → outcome dict
        self.time_to_fill_seconds: list[float] = []

    def clear_recovery(self):
        with self.lock:
            self.recovery = None


STATE = _State()
