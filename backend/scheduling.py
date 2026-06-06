"""Capacity-aware recovery planner (pure, no DB / no threads → easy to test).

When a slot is cancelled we don't just look for one replacement — we try to use
its whole duration well, in three tiers:

  TIER 1  Waitlist packing  — fill the freed minutes with waiting patients,
          newest-need first (3-day window), packing several short treatments
          into one long slot if no single long-treatment patient takes it.
  TIER 2  Pull-forward      — if capacity remains, pull a patient who is already
          booked LATER into the freed (earlier) slot. Nearby slots may move for
          any treatment that fits; weeks-ahead slots only for the SAME treatment
          (easier swap). Ranked by how much earlier they'd be seen (fairness),
          never by money.
  CASCADE Each pulled-forward patient frees their later slot, which we recover
          again (bounded depth) — one cancellation can improve several patients.

`plan_recovery` returns a RecoveryPlan whose properties are the scores:
utilization, € recovered, patients helped, cascade count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from backend.scoring import (
    MIN_TREATMENT_MIN, TREATMENT_MINUTES, TREATMENT_VALUE, CallStats, Patient,
    Slot, rank,
)

ReliabilityFn = Callable[..., float]
NEAR_DAYS = 14          # within this horizon a pull-forward may swap treatment types
MAX_CASCADE = 2         # bound the freed-slot recursion


@dataclass
class Booking:
    kind: str            # "waitlist" | "pull_forward"
    patient_id: int
    patient_name: str
    treatment: str
    minutes: int
    value_eur: int
    from_slot_id: Optional[int] = None   # the later slot freed by a pull-forward


@dataclass
class RecoveryPlan:
    slot_id: int
    capacity_min: int
    bookings: list[Booking] = field(default_factory=list)
    cascades: list["RecoveryPlan"] = field(default_factory=list)
    escalated: bool = False

    @property
    def filled_min(self) -> int:
        return sum(b.minutes for b in self.bookings)

    @property
    def recovered_eur(self) -> int:
        return sum(b.value_eur for b in self.bookings) + sum(c.recovered_eur for c in self.cascades)

    @property
    def utilization(self) -> float:
        return self.filled_min / self.capacity_min if self.capacity_min else 0.0

    @property
    def patients_helped(self) -> int:
        return len(self.bookings) + sum(c.patients_helped for c in self.cascades)

    @property
    def cascade_count(self) -> int:
        return sum(1 + c.cascade_count for c in self.cascades)


def _pull_forward_pool(slot, booked, capacity, now, used):
    """Booked-later patients who could move into the freed slot. Nearby slots may
    swap any fitting treatment; weeks-ahead only the same type. Sorted by the
    biggest earliness gain (fairness), not value."""
    near_cutoff = slot.start + timedelta(days=NEAR_DAYS)
    short_notice_slot = (slot.start - now).total_seconds() < 24 * 3600
    pool = []
    for bslot, patient in booked:
        if patient.id in used or not patient.consent_outbound:
            continue
        if short_notice_slot and not patient.short_notice_ok:
            continue                            # can't make the earlier slot on short notice
        if bslot.start <= slot.start:          # only pull patients booked LATER
            continue
        minutes = TREATMENT_MINUTES.get(bslot.type, bslot.duration_min)
        if minutes > capacity:
            continue
        same_type = bslot.type == slot.type
        if bslot.start > near_cutoff and not same_type:
            continue                            # weeks ahead ⇒ same treatment only
        days_earlier = (bslot.start - slot.start).days
        pool.append((days_earlier, bslot, patient, minutes,
                     TREATMENT_VALUE.get(bslot.type, bslot.value_eur)))
    pool.sort(key=lambda r: r[0], reverse=True)
    return pool


def plan_recovery(
    slot: Slot,
    waitlist: list[Patient],
    booked: list[tuple[Slot, Patient]],
    reliability: ReliabilityFn,
    *,
    call_stats: dict[int, CallStats] | None = None,
    now: Optional[datetime] = None,
    max_cascade: int = MAX_CASCADE,
    _used: set[int] | None = None,
) -> RecoveryPlan:
    now = now or datetime.now()
    used = _used if _used is not None else set()
    capacity = slot.duration_min
    plan = RecoveryPlan(slot_id=slot.id, capacity_min=capacity)

    # TIER 1 — waitlist packing
    while capacity >= MIN_TREATMENT_MIN:
        avail = [p for p in waitlist if p.id not in used]
        ranked, _, _ = rank(slot, avail, reliability, capacity_min=capacity,
                            call_stats_by_pid=call_stats, now=now)
        if not ranked:
            break
        top = ranked[0]
        plan.bookings.append(Booking(
            "waitlist", top.patient_id, top.name, top.treatment,
            top.treatment_minutes, TREATMENT_VALUE.get(top.treatment, 0)))
        used.add(top.patient_id)
        capacity -= top.treatment_minutes

    # TIER 2 — pull-forward (+ cascade on the slot each pulled patient vacates)
    while capacity >= MIN_TREATMENT_MIN:
        pool = _pull_forward_pool(slot, booked, capacity, now, used)
        if not pool:
            break
        _days, bslot, patient, minutes, value = pool[0]
        plan.bookings.append(Booking(
            "pull_forward", patient.id, patient.name, bslot.type, minutes, value,
            from_slot_id=bslot.id))
        used.add(patient.id)
        capacity -= minutes
        if max_cascade > 0:
            remaining_booked = [(s, p) for (s, p) in booked if s.id != bslot.id]
            sub = plan_recovery(
                bslot, waitlist, remaining_booked, reliability,
                call_stats=call_stats, now=now, max_cascade=max_cascade - 1, _used=used)
            if sub.bookings:
                plan.cascades.append(sub)

    plan.escalated = not plan.bookings
    return plan
