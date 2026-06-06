"""Filters + accept score + deadline-aware priority.

Implements §5.2–5.4 of PLAN.md. Pure functions, no DB, easy to unit test.
The reliability model is injected (see backend/reliability.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Callable, Optional


# --- domain dataclasses (DB rows decoded) ---

@dataclass
class Slot:
    id: int
    start: datetime
    duration_min: int
    type: str
    value_eur: int
    lead_days: int = 0  # booking horizon (for booked-slot no-show risk); 0 for fresh open slots


@dataclass
class Patient:
    id: int
    name: str
    phone: str
    age: int
    sms_opt_in: bool
    hypertension: bool
    diabetes: bool
    consent_outbound: bool
    short_notice_ok: bool
    preferred_window_start: time
    preferred_window_end: time
    needed_treatments: list[str]
    days_waiting: int
    attendance_history: list[int]
    last_offer_called_at: Optional[datetime]
    last_decline_at: Optional[datetime]
    last_declined_slot_type: Optional[str]


@dataclass
class Skip:
    patient_id: int
    name: str
    reason: str


@dataclass
class Ranked:
    patient_id: int
    name: str
    phone: str
    score: float
    answer_prob: float
    accept_score: float
    value_norm: float
    phase: str
    reason: str = ""
    days_waiting: int = 0
    window_match: bool = False
    attendance: tuple[int, int] = (0, 5)
    call_history: tuple[int, int, int] = (0, 0, 0)  # offers, answered, accepted (from log)


# --- decoders ---

def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def parse_patient(row) -> Patient:
    return Patient(
        id=row["id"],
        name=row["name"],
        phone=row["phone"],
        age=row["age"],
        sms_opt_in=bool(row["sms_opt_in"]),
        hypertension=bool(row["hypertension"]),
        diabetes=bool(row["diabetes"]),
        consent_outbound=bool(row["consent_outbound"]),
        short_notice_ok=bool(row["short_notice_ok"]),
        preferred_window_start=_parse_time(row["preferred_window_start"]),
        preferred_window_end=_parse_time(row["preferred_window_end"]),
        needed_treatments=json.loads(row["needed_treatments"] or "[]"),
        days_waiting=row["days_waiting"],
        attendance_history=json.loads(row["attendance_history"] or "[]"),
        last_offer_called_at=(
            datetime.fromisoformat(row["last_offer_called_at"])
            if row["last_offer_called_at"] else None
        ),
        last_decline_at=(
            datetime.fromisoformat(row["last_decline_at"])
            if row["last_decline_at"] else None
        ),
        last_declined_slot_type=row["last_declined_slot_type"],
    )


def parse_slot(row) -> Slot:
    return Slot(
        id=row["id"],
        start=datetime.fromisoformat(row["start_dt"] if "start_dt" in row.keys() else row["start"]),
        duration_min=row["duration_min"],
        type=row["type"],
        value_eur=row["value_eur"],
        lead_days=(row["lead_days"] if "lead_days" in row.keys() else 0),
    )


# --- phase ---

def phase_of(slot: Slot, now: datetime) -> str:
    """RELAXED / URGENT / CRITICAL / UNRECOVERABLE per §5.4."""
    minutes_left = (slot.start - now).total_seconds() / 60.0
    if minutes_left < 20:
        return "UNRECOVERABLE"
    if minutes_left < 120:
        return "CRITICAL"
    if minutes_left < 24 * 60:
        return "URGENT"
    return "RELAXED"


# --- filters (§5.3) ---

def apply_hard_filters(
    patient: Patient,
    slot: Slot,
    now: datetime,
    *,
    offers_this_week: int = 0,
    only_feasible: bool = False,
) -> Optional[str]:
    """Return a skip reason string, or None if patient passes all filters.

    `only_feasible=True` relaxes the cooldown rule per §5.3 (allow, flag).
    Weekly cap is enforced by the caller; we accept the count.
    """
    if not patient.consent_outbound:
        return "No consent for outbound calls"
    if slot.type not in patient.needed_treatments:
        return "Treatment mismatch"

    minutes_left = (slot.start - now).total_seconds() / 60.0
    if not patient.short_notice_ok and minutes_left < 24 * 60:
        return "Cannot make it on short notice"

    if patient.last_offer_called_at:
        hours_since_offer = (now - patient.last_offer_called_at).total_seconds() / 3600.0
        if hours_since_offer < 72 and not only_feasible:
            return "Cooldown (called recently)"

    if patient.last_decline_at and patient.last_declined_slot_type == slot.type:
        days_since = (now - patient.last_decline_at).days
        if days_since < 7:
            return "Recently declined similar"

    if offers_this_week >= 2:
        return "Weekly contact cap"

    return None


# --- accept score (§5.2) ---

def accept_score(patient: Patient, slot: Slot) -> float:
    s = 0.5
    start_t = slot.start.time()
    if patient.preferred_window_start <= start_t <= patient.preferred_window_end:
        s += 0.25
    if slot.type in patient.needed_treatments:
        s += 0.15
    s += min(patient.days_waiting / 60.0, 0.15)
    if (
        patient.last_decline_at
        and patient.last_declined_slot_type == slot.type
        and (datetime.now() - patient.last_decline_at).days < 7
    ):
        s -= 0.30
    return max(0.05, min(0.95, s))


# --- learned signal from the call log (§5.2 upgrade) ---
#
# The accept score above and the reliability model are *priors*. The system
# logs every offer call's outcome, so each patient also has an empirical
# pick-up rate (P answer) and acceptance rate (P accept | answered). We shrink
# the prior toward those observed rates — the engine learns who actually
# responds, from its own history, and stays cold-start safe.

@dataclass
class CallStats:
    """Per-patient outbound-offer history, from the calls table."""
    offers: int = 0       # completed offer calls
    answered: int = 0     # picked up (booked / declined / callback)
    accepted: int = 0     # booked

# Prior strength = pseudo-observations: how many real calls it takes for the
# observed rate to outweigh the prior. Higher ⇒ trust the prior longer.
ANSWER_PRIOR_STRENGTH = 4.0
ACCEPT_PRIOR_STRENGTH = 4.0


def _bayes(prior_mean: float, prior_strength: float, successes: int, trials: int) -> float:
    """Shrink the prior toward the observed rate as evidence accrues.
    trials == 0 ⇒ returns the prior unchanged (cold-start safe)."""
    return (prior_mean * prior_strength + successes) / (prior_strength + trials)


def learned_answer(model_p: float, stats: Optional["CallStats"]) -> float:
    """Blend the model's P(show) prior with the patient's real pick-up rate."""
    if not stats or stats.offers == 0:
        return model_p
    return _bayes(model_p, ANSWER_PRIOR_STRENGTH, stats.answered, stats.offers)


def learned_accept(heuristic_p: float, stats: Optional["CallStats"]) -> float:
    """Blend the §5.2 heuristic prior with the patient's real acceptance rate."""
    if not stats or stats.answered == 0:
        return heuristic_p
    return _bayes(heuristic_p, ACCEPT_PRIOR_STRENGTH, stats.accepted, stats.answered)


# --- final priority (§5.4) ---

def deadline_priority(
    answer: float, accept: float, value_norm: float, hours_left: float, days_waiting: int
) -> tuple[float, float]:
    """Return (score, urgency_mode). Starvation guard applied here."""
    urgency = max(0.0, min(1.0, (24.0 - hours_left) / 24.0))
    score = (answer ** (1 + urgency)) * accept * (value_norm ** (1 - 0.7 * urgency))
    if days_waiting > 30:
        score *= 1.5
    return score, urgency


# --- top-level: rank ---

ReliabilityFn = Callable[..., float]  # (Patient, lead_days: float) -> P(showed)


def rank(
    slot: Slot,
    patients: list[Patient],
    reliability: ReliabilityFn,
    *,
    max_value_eur: int = 600,
    exclude_ids: set[int] | None = None,
    offers_this_week_by_pid: dict[int, int] | None = None,
    call_stats_by_pid: dict[int, "CallStats"] | None = None,
    now: Optional[datetime] = None,
    top_k: int = 5,
) -> tuple[list[Ranked], list[Skip], str]:
    """Filter then score then sort. Returns (ranked, skipped, phase)."""
    now = now or datetime.now()
    exclude_ids = exclude_ids or set()
    offers = offers_this_week_by_pid or {}
    call_stats = call_stats_by_pid or {}
    phase = phase_of(slot, now)

    if phase == "UNRECOVERABLE":
        return [], [Skip(0, "—", "Slot is within the unrecoverable window (<20 min)")], phase

    # First pass: assume there are other feasible candidates.
    skips: list[Skip] = []
    accepted: list[Patient] = []
    for p in patients:
        if p.id in exclude_ids:
            continue
        reason = apply_hard_filters(
            p, slot, now, offers_this_week=offers.get(p.id, 0), only_feasible=False
        )
        if reason:
            skips.append(Skip(p.id, p.name, reason))
        else:
            accepted.append(p)

    # Cooldown rescue: if nobody passed, retry cooldown'd patients with relax flag.
    rescued_ids: set[int] = set()
    if not accepted:
        keep_skips: list[Skip] = []
        for skip in skips:
            if skip.reason == "Cooldown (called recently)":
                p = next((x for x in patients if x.id == skip.patient_id), None)
                if p:
                    rescued_ids.add(p.id)
                    accepted.append(p)
            else:
                keep_skips.append(skip)
        skips = keep_skips

    ranked: list[Ranked] = []
    hours_left = (slot.start - now).total_seconds() / 3600.0
    lead_days = max(0.0, hours_left / 24.0)  # a same-day refill ⇒ near-zero booking lead
    for p in accepted:
        stats = call_stats.get(p.id)
        # priors (model + heuristic) shrunk toward this patient's real call record
        ans = learned_answer(reliability(p, lead_days), stats)
        acc = learned_accept(accept_score(p, slot), stats)
        vnorm = slot.value_eur / max(max_value_eur, 1)
        score, _ = deadline_priority(ans, acc, vnorm, hours_left, p.days_waiting)
        start_t = slot.start.time()
        in_window = p.preferred_window_start <= start_t <= p.preferred_window_end
        attended = sum(p.attendance_history)
        total = max(len(p.attendance_history), 1)
        ch = (stats.offers, stats.answered, stats.accepted) if stats else (0, 0, 0)
        ranked.append(Ranked(
            patient_id=p.id,
            name=p.name + (" (cooldown override)" if p.id in rescued_ids else ""),
            phone=p.phone,
            score=score,
            answer_prob=ans,
            accept_score=acc,
            value_norm=vnorm,
            phase=phase,
            days_waiting=p.days_waiting,
            window_match=in_window,
            attendance=(attended, total),
            call_history=ch,
        ))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k], skips, phase
