"""Recovery state machine (§6.4) — now capacity-aware.

Strictly sequential dialing. UNIQUE(slot_id) on recovery_attempts + an in-memory
guard make it idempotent under duplicate cancel events. Runs in a background
thread so the FastAPI request returns immediately.

Filling a freed slot:
  TIER 1  waitlist — book the best waiting patient whose treatment fits the open
          time (capacity-aware → a long slot can take a shorter treatment).
  TIER 2  pull-forward — if no waiting patient takes it, pull a patient booked
          LATER into the earlier slot, then recover the slot they vacate.
  PACKING leftover time after a short booking is spun off as a new slot and
          recovered too. Both leftover + pull-forward use recursive
          trigger_recovery (bounded by MAX_RECOVERY_DEPTH).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, time as dtime, timedelta
from typing import Optional

from backend import reliability, repo
from backend.fonio_client import FonioClient, build_client
from backend.reasons import reason_for
from backend.scoring import MIN_TREATMENT_MIN, TREATMENT_VALUE, rank
from backend.state import STATE, CurrentRecovery

LOG = logging.getLogger("orchestrator")

_LOCK = threading.RLock()
_in_flight: set[int] = set()
_patient_locks: set[int] = set()  # §8: lock a patient while a call for them is in flight

WEBHOOK_TIMEOUT_S = int(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "90"))
MAX_RECOVERY_DEPTH = int(os.environ.get("MAX_RECOVERY_DEPTH", "3"))
_client: FonioClient | None = None


def _client_lazy() -> FonioClient:
    global _client
    if _client is None:
        _client = build_client()
    return _client


# --- call-window check (§6.4: no calls before 08:00 / after 19:00) ---

def in_call_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    start = _parse_hhmm(os.environ.get("CALL_WINDOW_START", "08:00"))
    end = _parse_hhmm(os.environ.get("CALL_WINDOW_END", "19:00"))
    return start <= now.time() <= end


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


# --- public entrypoint ---

def trigger_recovery(slot_id: int, depth: int = 0) -> dict:
    """Idempotent. Returns status JSON. depth>0 = a cascade/leftover recovery."""
    with _LOCK:
        if slot_id in _in_flight:
            return {"ok": False, "reason": "already recovering"}
        try:
            repo.create_recovery_attempt(slot_id)
        except sqlite3.IntegrityError:
            return {"ok": False, "reason": "duplicate cancellation event (idempotent)"}
        _in_flight.add(slot_id)

    repo.set_slot_status(slot_id, "recovering")
    if depth == 0:
        STATE.recovery = CurrentRecovery(slot_id=slot_id, phase="—", started_at=datetime.now())
    threading.Thread(target=_run, args=(slot_id, depth), daemon=True).start()
    return {"ok": True}


# --- helpers ---

def _publish(depth: int, phase: str, ranked, skipped, tried: set[int]) -> None:
    if depth != 0:
        return
    cands = [{
        "patient_id": r.patient_id, "name": r.name, "phone": r.phone,
        "score": round(r.score, 4),
        "breakdown": {"answer_prob": round(r.answer_prob, 3),
                      "accept_score": round(r.accept_score, 3),
                      "value_norm": round(r.value_norm, 3), "phase": r.phase},
        "reason": reason_for(r),
    } for r in ranked]
    skips = [{"patient_id": s.patient_id, "name": s.name, "reason": s.reason} for s in skipped]
    with STATE.lock:
        if STATE.recovery is not None:
            STATE.recovery.phase = phase
            STATE.recovery.candidates = cands
            STATE.recovery.skipped = skips
            STATE.recovery.tried_patient_ids = list(tried)


def _vars(patient, slot) -> dict:
    return {
        "patient_name": patient.name.split()[0],
        "slot_time": slot.start.strftime("%H:%M"),
        "treatment": slot.type,
        "practice_name": os.environ.get("PRACTICE_NAME", "Smile Dental"),
    }


def _call(patient, slot, slot_id: int, attempt_id, depth: int, *, pull_forward=False) -> str:
    """Place one offer call and return its outcome (booked/declined/voicemail/
    callback), 'refused' if fonio wouldn't take the trigger, or 'busy' if another
    in-flight recovery already holds this patient (atomic check-and-claim closes
    the TOCTOU: two concurrent recoveries can never call the same patient)."""
    with _LOCK:
        if patient.id in _patient_locks:
            LOG.info("patient %s already in flight in another recovery; skipping", patient.id)
            return "busy"
        _patient_locks.add(patient.id)
    try:
        if depth == 0:
            with STATE.lock:
                if STATE.recovery is not None:
                    STATE.recovery.current_patient_id = patient.id
                    STATE.recovery.current_patient_name = (
                        patient.name + (" (pull-forward)" if pull_forward else ""))
                    STATE.recovery.current_started_at = datetime.now()
        tr = _client_lazy().trigger_offer(
            slot_id=slot_id, patient_id=patient.id, phone=patient.phone,
            variables=_vars(patient, slot))
        if not tr.accepted:
            LOG.warning("fonio refused call for patient %s: %s", patient.id, tr.error)
            return "refused"
        repo.record_offer_called(patient.id)
        repo.log_call(
            fonio_call_id=tr.fonio_call_id, recovery_attempt_id=attempt_id,
            patient_id=patient.id, slot_id=slot_id, direction="outbound",
            outcome=None, summary=None)
        outcome, summary = _wait_for_outcome(tr.fonio_call_id, timeout=WEBHOOK_TIMEOUT_S)
        repo.update_call_outcome(tr.fonio_call_id, outcome, summary or "")
        return outcome
    finally:
        with _LOCK:
            _patient_locks.discard(patient.id)


def _booked(slot, slot_id, patient, treatment, minutes, value, started_wall, depth) -> bool:
    """Atomic compare-and-set commit. Returns True only if THIS recovery won the
    slot (it was still 'recovering'); on False a concurrent winner / duplicate
    already filled it, so we skip booking + attendance — no double-book."""
    if not repo.fill_slot(slot_id, patient.id, treatment, minutes, value):
        LOG.warning("lost race: slot %s already filled, skipping double-book (patient %s)",
                    slot_id, patient.id)
        return False
    repo.finish_recovery(slot_id, "filled", patient.id)
    repo.push_attendance(patient.id, 1)
    if depth == 0:
        STATE.time_to_fill_seconds.append(time.time() - started_wall)
    LOG.info("slot %s filled by patient %s (%s, %dmin)", slot_id, patient.id, treatment, minutes)
    return True


def _run(slot_id: int, depth: int = 0) -> None:
    started_wall = time.time()
    booked_someone = False
    try:
        slot = repo.get_slot(slot_id)
        if not slot:
            return
        if not in_call_window():
            LOG.info("outside call window; escalating slot %s", slot_id)
            repo.set_slot_status(slot_id, "escalated")
            repo.finish_recovery(slot_id, "escalated_window", None)
            return

        attempt = repo.recovery_attempt_for(slot_id)
        attempt_id = attempt["id"] if attempt else None
        capacity = slot.duration_min
        tried: set[int] = set()

        # ---- TIER 1: waitlist (capacity-aware) ----
        while capacity >= MIN_TREATMENT_MIN:
            patients = repo.waitlist_patients()   # true waitlist (no current booking)
            ranked, skipped, phase = rank(
                slot, patients, reliability.predict,
                exclude_ids=tried | _patient_locks,
                offers_this_week_by_pid=repo.offers_this_week_by_pid(),
                call_stats_by_pid=repo.call_stats_by_pid(),
                capacity_min=capacity,
            )
            _publish(depth, phase, ranked, skipped, tried)
            if phase == "UNRECOVERABLE":
                break
            if not ranked:
                break
            top = ranked[0]
            patient = repo.get_patient(top.patient_id)
            if patient is None:
                tried.add(top.patient_id); continue
            outcome = _call(patient, slot, slot_id, attempt_id, depth)
            if outcome in ("refused", "busy"):
                tried.add(patient.id); continue
            if outcome == "booked":
                value = TREATMENT_VALUE.get(top.treatment, slot.value_eur)
                if not _booked(slot, slot_id, patient, top.treatment, top.treatment_minutes,
                               value, started_wall, depth):
                    # lost the slot to a concurrent winner — do NOT book/pack/attend.
                    tried.add(patient.id)
                    return
                booked_someone = True
                leftover = capacity - top.treatment_minutes
                if leftover >= MIN_TREATMENT_MIN and depth < MAX_RECOVERY_DEPTH:
                    new_id = repo.create_open_slot(
                        slot.start + timedelta(minutes=top.treatment_minutes),
                        leftover, slot.type, slot.value_eur)
                    LOG.info("packing: %dmin leftover on slot %s → recovering slot %s",
                             leftover, slot_id, new_id)
                    trigger_recovery(new_id, depth + 1)
                return
            if outcome == "declined":
                repo.record_decline(patient.id, slot.type)
            tried.add(patient.id)
            time.sleep(0.3)

        # ---- TIER 2: pull-forward (only reached if no waitlist patient took it) ----
        if not booked_someone:
            while capacity >= MIN_TREATMENT_MIN:
                cands = repo.pull_forward_candidates(
                    slot, capacity, exclude_pids=tried | _patient_locks)
                if not cands:
                    break
                bslot, patient, treatment, minutes, value = cands[0]
                outcome = _call(patient, slot, slot_id, attempt_id, depth, pull_forward=True)
                if outcome in ("refused", "busy"):
                    tried.add(patient.id); continue
                if outcome == "booked":
                    # ATOMIC move: fill THIS slot + free their later slot in one txn.
                    # If we lost the slot to a concurrent winner, nothing is written —
                    # the patient is never double-booked.
                    if not repo.pull_forward_commit(bslot.id, slot_id, patient.id,
                                                    treatment, minutes, value):
                        LOG.warning("lost race: pull-forward slot %s already filled, "
                                    "skipping (patient %s)", slot_id, patient.id)
                        tried.add(patient.id)
                        continue
                    repo.finish_recovery(slot_id, "filled", patient.id)
                    repo.push_attendance(patient.id, 1)
                    if depth == 0:
                        STATE.time_to_fill_seconds.append(time.time() - started_wall)
                    booked_someone = True
                    LOG.info("pull-forward: patient %s moved into slot %s, freeing slot %s",
                             patient.id, slot_id, bslot.id)
                    if depth < MAX_RECOVERY_DEPTH:
                        trigger_recovery(bslot.id, depth + 1)   # cascade
                    return
                tried.add(patient.id)
                time.sleep(0.3)

        if not booked_someone:
            LOG.info("slot %s exhausted/unrecoverable", slot_id)
            status = "unrecoverable" if (slot.start - datetime.now()) < timedelta(minutes=20) else "escalated"
            repo.set_slot_status(slot_id, status)
            repo.finish_recovery(slot_id, f"{status}_exhausted", None)
    finally:
        with _LOCK:
            _in_flight.discard(slot_id)
        if depth == 0:
            with STATE.lock:
                STATE.recovery = None


def _wait_for_outcome(fonio_call_id: str, *, timeout: int) -> tuple[str, str]:
    """Polls STATE.webhook_events. Returns (outcome, summary)."""
    from backend.outcome_parser import parse_outcome
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = STATE.webhook_events.pop(fonio_call_id, None)
        if ev is not None:
            outcome = parse_outcome(ev.get("summary"), disconnect_reason=ev.get("disconnectReason"))
            return outcome, ev.get("summary", "") or ""
        time.sleep(0.25)
    return "voicemail", "[orchestrator] webhook timeout"
