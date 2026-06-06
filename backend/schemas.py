"""Pydantic schemas — these define the §4 wire contract.

Track A serves, Track B consumes. Do not break compatibility without
telling the other pair.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---- /rank ----

class SlotIn(BaseModel):
    id: int
    start: str  # ISO 8601
    duration_min: int
    type: str
    value_eur: int


class RankRequest(BaseModel):
    slot: SlotIn
    exclude_patient_ids: list[int] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    answer_prob: float
    accept_score: float
    value_norm: float
    phase: Literal["RELAXED", "URGENT", "CRITICAL", "UNRECOVERABLE"]


class Candidate(BaseModel):
    patient_id: int
    name: str
    phone: str
    score: float
    breakdown: ScoreBreakdown
    reason: str


class Skipped(BaseModel):
    patient_id: int
    name: str
    reason: str


class RankResponse(BaseModel):
    candidates: list[Candidate]
    skipped: list[Skipped]
    phase: Literal["RELAXED", "URGENT", "CRITICAL", "UNRECOVERABLE"]


# ---- /outcome ----

Outcome = Literal["booked", "declined", "voicemail", "callback", "timeout"]


class OutcomeRequest(BaseModel):
    slot_id: int
    patient_id: int
    outcome: Outcome
    fonio_call_id: Optional[str] = None
    summary: Optional[str] = None


class OutcomeResponse(BaseModel):
    ok: bool
    slot_status: str


# ---- dashboard ----

class DashboardSlot(BaseModel):
    id: int
    start: str
    duration_min: int
    type: str
    value_eur: int
    status: str
    booked_patient_name: Optional[str] = None
    no_show_risk: Optional[float] = None  # 0..1 (1 - answer_prob)


class RecoveryState(BaseModel):
    slot_id: Optional[int] = None
    phase: Optional[str] = None
    current_patient_id: Optional[int] = None
    current_patient_name: Optional[str] = None
    current_started_at: Optional[str] = None
    candidates: list[Candidate] = Field(default_factory=list)
    skipped: list[Skipped] = Field(default_factory=list)
    tried_patient_ids: list[int] = Field(default_factory=list)


class Metrics(BaseModel):
    refill_rate_pct: float
    eur_recovered: int
    avg_time_to_fill_seconds: Optional[float]
    outcomes: dict[str, int]


class DashboardState(BaseModel):
    now: str
    schedule: list[DashboardSlot]
    recovery: RecoveryState
    metrics: Metrics
