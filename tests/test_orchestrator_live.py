"""Live-orchestrator integration tests: drive the real recovery threads (with a
stub fonio client that answers instantly) and assert packing + pull-forward +
cascade actually happen end to end."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("CALL_WINDOW_START", "00:00")
os.environ.setdefault("CALL_WINDOW_END", "23:59")

from backend import orchestrator, reliability, repo
from backend.db import connect, reset_db
from backend.fonio_client import TriggerResult
from backend.state import STATE

VAL = {"cleaning": 80, "checkup": 60, "filling": 150, "crown": 600}


class StubAccept:
    """Answers every call immediately; books unless the patient is in `decline`."""

    def __init__(self, decline: set[int] | None = None):
        self.decline = decline or set()

    def trigger_offer(self, *, slot_id, patient_id, phone, variables):
        cid = f"stub-{patient_id}-{uuid.uuid4().hex[:6]}"
        token = "OUTCOME_DECLINED" if patient_id in self.decline else "OUTCOME_BOOKED"
        STATE.webhook_events[cid] = {"id": cid, "summary": token}
        return TriggerResult(fonio_call_id=cid, accepted=True)

    def trigger_confirmation(self, **kw):
        return self.trigger_offer(**kw)


def _patient(c, pid, name, treatments, consent=1, short=1):
    c.execute(
        """INSERT INTO patients(id,name,phone,age,sms_opt_in,hypertension,diabetes,
             consent_outbound,short_notice_ok,preferred_window_start,preferred_window_end,
             needed_treatments,days_waiting,attendance_history,last_offer_called_at,
             last_decline_at,last_declined_slot_type)
           VALUES (?,?,?,35,1,0,0,?,?,'08:00','19:00',?,?,?,NULL,NULL,NULL)""",
        (pid, name, f"+430{pid:06d}", consent, short,
         json.dumps(treatments), 15, json.dumps([1, 1, 1, 1, 1])))


def _slot(c, sid, ttype, minutes, when, status="booked", pid=None):
    c.execute(
        "INSERT INTO slots(id,start_dt,duration_min,type,value_eur,status,booked_patient_id) "
        "VALUES (?,?,?,?,?,?,?)",
        (sid, when.isoformat(), minutes, ttype, VAL[ttype], status, pid))


def _wait(pred, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(autouse=True)
def _fresh():
    deadline = time.time() + 5          # let any prior recovery threads finish (Win file lock)
    while orchestrator._in_flight and time.time() < deadline:
        time.sleep(0.05)
    reset_db()
    reliability.load()
    orchestrator._in_flight.clear()
    orchestrator._patient_locks.clear()
    STATE.webhook_events.clear()
    STATE.recovery = None
    yield


def _filled():
    conn = connect()
    try:
        return conn.execute("SELECT * FROM slots WHERE status='filled'").fetchall()
    finally:
        conn.close()


def test_packing_fills_leftover_capacity_live():
    """A 60-min crown slot, two cleaning patients on the waitlist, no crown taker
    → Tier 1 books one cleaning, the 30-min leftover is recovered and books the
    other. Two filled appointments from one cancellation."""
    conn = connect()
    try:
        _patient(conn, 1, "A", ["cleaning"])
        _patient(conn, 2, "B", ["cleaning"])
        _slot(conn, 1, "crown", 60, datetime.now() + timedelta(hours=5), status="cancelled")
    finally:
        conn.close()
    orchestrator._client = StubAccept()
    orchestrator.trigger_recovery(1)

    assert _wait(lambda: len(_filled()) >= 2), "packing should produce two filled slots"
    filled = _filled()
    pids = {r["booked_patient_id"] for r in filled}
    assert pids == {1, 2}                                   # both patients booked, no double-book
    assert all(r["type"] == "cleaning" for r in filled)     # crown time repurposed to cleanings


def test_pull_forward_and_cascade_live():
    """Freed crown slot today; the only waitlister can't do short notice; a patient
    is booked for a crown 21 days out → pull them forward, then the freed far slot
    cascades to the short-notice patient (who's fine 21 days out)."""
    conn = connect()
    try:
        _patient(conn, 1, "Greta", ["crown"], short=0)     # can't do today
        _patient(conn, 2, "Frank", ["crown"], short=1)     # booked later, can move up
        _slot(conn, 1, "crown", 60, datetime.now() + timedelta(hours=5), status="cancelled")
        _slot(conn, 2, "crown", 60, datetime.now() + timedelta(days=21), status="booked", pid=2)
    finally:
        conn.close()
    orchestrator._client = StubAccept()
    orchestrator.trigger_recovery(1)

    # Frank pulled into slot 1; slot 2 freed then cascaded to Greta.
    assert _wait(lambda: repo.slot_status(1) == "filled" and repo.slot_status(2) == "filled")
    conn = connect()
    try:
        s1 = conn.execute("SELECT booked_patient_id FROM slots WHERE id=1").fetchone()
        s2 = conn.execute("SELECT booked_patient_id FROM slots WHERE id=2").fetchone()
    finally:
        conn.close()
    assert s1["booked_patient_id"] == 2     # Frank moved earlier
    assert s2["booked_patient_id"] == 1     # Greta took the freed later slot (cascade)
