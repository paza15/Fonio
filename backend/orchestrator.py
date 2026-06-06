"""Recovery state machine (§6.4).

Strictly sequential dialing. UNIQUE(slot_id) on recovery_attempts +
in-memory guard make it idempotent under duplicate cancel events. Runs in a
background thread so the FastAPI request returns immediately.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, time as dtime
from typing import Optional

import httpx

from backend import repo, reliability
from backend.fonio_client import FonioClient, build_client
from backend.outcome_parser import parse_outcome
from backend.reasons import reason_for
from backend.scoring import rank
from backend.state import STATE, CurrentRecovery

LOG = logging.getLogger("orchestrator")

_LOCK = threading.RLock()
_in_flight: set[int] = set()
_patient_locks: set[int] = set()  # §8: lock patient while a call is in flight

WEBHOOK_TIMEOUT_S = int(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "90"))
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

def trigger_recovery(slot_id: int) -> dict:
    """Idempotent. Returns status JSON for the API.

    DB-level guarantee: UNIQUE(slot_id) on recovery_attempts.
    Process-level guarantee: _in_flight set.
    """
    with _LOCK:
        if slot_id in _in_flight:
            return {"ok": False, "reason": "already recovering"}
        try:
            repo.create_recovery_attempt(slot_id)
        except sqlite3.IntegrityError:
            return {"ok": False, "reason": "duplicate cancellation event (idempotent)"}
        _in_flight.add(slot_id)

    repo.set_slot_status(slot_id, "recovering")
    STATE.recovery = CurrentRecovery(
        slot_id=slot_id, phase="—",
        started_at=datetime.now(),
    )
    threading.Thread(target=_run, args=(slot_id,), daemon=True).start()
    return {"ok": True}


def _run(slot_id: int) -> None:
    started_wall = time.time()
    try:
        slot = repo.get_slot(slot_id)
        if not slot:
            LOG.error("slot %s vanished", slot_id)
            return

        if not in_call_window():
            LOG.info("outside call window; escalating slot %s", slot_id)
            repo.set_slot_status(slot_id, "escalated")
            repo.finish_recovery(slot_id, "escalated_window", None)
            return

        tried: set[int] = set()
        rec_attempt = repo.recovery_attempt_for(slot_id)
        attempt_id = rec_attempt["id"] if rec_attempt else None

        while True:
            patients = repo.all_patients()
            offers = repo.offers_this_week_by_pid()
            ranked, skipped, phase = rank(
                slot, patients, reliability.predict,
                exclude_ids=tried | _patient_locks,
                offers_this_week_by_pid=offers,
            )
            # attach reasons
            cand_for_state = []
            for r in ranked:
                cand_for_state.append({
                    "patient_id": r.patient_id, "name": r.name, "phone": r.phone,
                    "score": round(r.score, 4),
                    "breakdown": {
                        "answer_prob": round(r.answer_prob, 3),
                        "accept_score": round(r.accept_score, 3),
                        "value_norm": round(r.value_norm, 3),
                        "phase": r.phase,
                    },
                    "reason": reason_for(r),
                })
            skipped_for_state = [
                {"patient_id": s.patient_id, "name": s.name, "reason": s.reason}
                for s in skipped
            ]
            with STATE.lock:
                if STATE.recovery is not None:
                    STATE.recovery.phase = phase
                    STATE.recovery.candidates = cand_for_state
                    STATE.recovery.skipped = skipped_for_state
                    STATE.recovery.tried_patient_ids = list(tried)

            if phase == "UNRECOVERABLE":
                LOG.info("slot %s unrecoverable", slot_id)
                repo.set_slot_status(slot_id, "unrecoverable")
                repo.finish_recovery(slot_id, "unrecoverable", None)
                return
            if not ranked:
                LOG.info("slot %s exhausted", slot_id)
                repo.set_slot_status(slot_id, "escalated")
                repo.finish_recovery(slot_id, "escalated_exhausted", None)
                return

            top = ranked[0]
            patient = repo.get_patient(top.patient_id)
            if patient is None:
                tried.add(top.patient_id); continue

            with _LOCK:
                _patient_locks.add(patient.id)

            # Mark current candidate on the dashboard
            with STATE.lock:
                if STATE.recovery is not None:
                    STATE.recovery.current_patient_id = patient.id
                    STATE.recovery.current_patient_name = patient.name
                    STATE.recovery.current_started_at = datetime.now()

            tr = _client_lazy().trigger_offer(
                slot_id=slot_id, patient_id=patient.id, phone=patient.phone,
                variables={
                    "patient_name": patient.name.split()[0],
                    "slot_time": slot.start.strftime("%H:%M"),
                    "treatment": slot.type,
                    "practice_name": os.environ.get("PRACTICE_NAME", "Smile Dental"),
                },
            )
            if not tr.accepted:
                LOG.warning("fonio refused call for patient %s: %s", patient.id, tr.error)
                with _LOCK: _patient_locks.discard(patient.id)
                tried.add(patient.id); continue

            repo.record_offer_called(patient.id)
            repo.log_call(
                fonio_call_id=tr.fonio_call_id,
                recovery_attempt_id=attempt_id,
                patient_id=patient.id, slot_id=slot_id,
                direction="outbound", outcome=None, summary=None,
            )

            outcome, summary = _wait_for_outcome(tr.fonio_call_id, timeout=WEBHOOK_TIMEOUT_S)
            repo.update_call_outcome(tr.fonio_call_id, outcome, summary or "")
            with _LOCK: _patient_locks.discard(patient.id)

            if outcome == "booked":
                repo.set_slot_status(slot_id, "filled", booked_patient_id=patient.id)
                repo.finish_recovery(slot_id, "filled", patient.id)
                repo.push_attendance(patient.id, 1)
                STATE.time_to_fill_seconds.append(time.time() - started_wall)
                LOG.info("slot %s filled by patient %s", slot_id, patient.id)
                return
            if outcome == "declined":
                repo.record_decline(patient.id, slot.type)
            tried.add(patient.id)
            time.sleep(0.5)
    finally:
        with _LOCK:
            _in_flight.discard(slot_id)
        with STATE.lock:
            STATE.recovery = None


def _wait_for_outcome(fonio_call_id: str, *, timeout: int) -> tuple[str, str]:
    """Polls STATE.webhook_events. Returns (outcome, summary)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = STATE.webhook_events.pop(fonio_call_id, None)
        if ev is not None:
            outcome = parse_outcome(
                ev.get("summary"),
                disconnect_reason=ev.get("disconnectReason"),
            )
            return outcome, ev.get("summary", "") or ""
        time.sleep(0.25)
    return "voicemail", "[orchestrator] webhook timeout"
