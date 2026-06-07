"""Proactive no-show PREVENTION — put the §5.1 model to work before the chair empties.

Instead of only reacting to cancellations, sweep upcoming booked appointments,
predict each one's no-show risk, and confirmation-call the risky ones ahead of time:

  "yes, I'm coming"  → mark confirmed (risk neutralised)
  "can't make it"    → cancel EARLY and recover now → maximum lead time, best refill
  no answer          → mark at-risk, leave for a human (never auto-cancel, §8)

This is the difference between "fill the gap" and "stop the gap happening."
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from backend import reliability, repo
from backend.fonio_client import FonioClient, build_client
from backend.state import STATE

LOG = logging.getLogger("prevention")

RISK_THRESHOLD = float(os.environ.get("PREVENTION_RISK_THRESHOLD", "0.35"))
HORIZON_HOURS = int(os.environ.get("PREVENTION_HORIZON_HOURS", "48"))
CONFIRM_TIMEOUT_S = int(os.environ.get("CONFIRM_TIMEOUT_SECONDS", "90"))

_client: FonioClient | None = None


def _client_lazy() -> FonioClient:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def at_risk_slots(horizon_hours: int = HORIZON_HOURS, threshold: float = RISK_THRESHOLD,
                  now: Optional[datetime] = None) -> list[tuple]:
    """Upcoming booked slots whose predicted no-show risk ≥ threshold, riskiest first.
    Returns [(slot_id, patient, risk), ...]."""
    now = now or datetime.now()
    until = now + timedelta(hours=horizon_hours)
    out = []
    for r in repo.upcoming_booked(now, until):
        patient = repo.get_patient(r["booked_patient_id"])
        if not patient:
            continue
        risk = 1.0 - reliability.predict(patient, lead_days=r.get("lead_days") or 0)
        if risk >= threshold:
            out.append((r["id"], patient, risk))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def _wait_confirmation(call_id: str, timeout: int) -> str:
    """Poll for the confirmation webhook → 'confirmed' | 'cancel' | 'noanswer'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = STATE.webhook_events.pop(call_id, None)
        if ev is not None:
            s = (ev.get("summary") or "").upper()
            if "CANCEL" in s or "NOT COMING" in s or "DECLINED" in s:
                return "cancel"
            if "CONFIRM" in s or "COMING" in s or "BOOKED" in s:
                return "confirmed"
            return "noanswer"
        time.sleep(0.25)
    return "noanswer"


def confirm_slot(slot_id: int, patient) -> str:
    tr = _client_lazy().trigger_confirmation(
        slot_id=slot_id, patient_id=patient.id, phone=patient.phone,
        variables={"patient_name": patient.name.split()[0]})
    if not tr.accepted:
        return "noanswer"
    repo.log_call(
        fonio_call_id=tr.fonio_call_id, recovery_attempt_id=None, patient_id=patient.id,
        slot_id=slot_id, to_number=patient.phone, direction="confirmation",
        outcome=None, summary=None)
    out = _wait_confirmation(tr.fonio_call_id, CONFIRM_TIMEOUT_S)
    repo.update_call_outcome(tr.fonio_call_id, out, "")
    return out


def run_sweep(horizon_hours: int = HORIZON_HOURS, threshold: float = RISK_THRESHOLD,
              now: Optional[datetime] = None) -> list[dict]:
    """Confirmation-call every at-risk slot and route the outcome. Returns a log."""
    from backend import orchestrator
    results = []
    for slot_id, patient, risk in at_risk_slots(horizon_hours, threshold, now):
        outcome = confirm_slot(slot_id, patient)
        if outcome == "cancel":
            repo.cancel_slot(slot_id)
            repo.set_confirmation(slot_id, "cancelled")
            orchestrator.trigger_recovery(slot_id)          # recover EARLY, max lead time
            action = "cancelled early → recovering"
        elif outcome == "confirmed":
            repo.set_confirmation(slot_id, "confirmed")
            action = "confirmed"
        else:
            repo.set_confirmation(slot_id, "at_risk")
            action = "at-risk → notify human"
        LOG.info("prevention: slot %s patient %s risk %.2f → %s", slot_id, patient.id, risk, action)
        results.append({"slot_id": slot_id, "patient_id": patient.id, "patient": patient.name,
                        "risk": round(risk, 3), "outcome": outcome, "action": action})
    return results


def run_sweep_async(**kwargs) -> None:
    threading.Thread(target=run_sweep, kwargs=kwargs, daemon=True).start()
