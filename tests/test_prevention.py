"""Proactive no-show prevention: risk selection + the confirmation sweep routing
(confirm / early-cancel→recover / at-risk)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("CALL_WINDOW_START", "00:00")
os.environ.setdefault("CALL_WINDOW_END", "23:59")

from backend import orchestrator, prevention, reliability, repo
from backend.db import connect, reset_db
from backend.fonio_client import MockFonioClient
from backend.state import STATE

VAL = {"cleaning": 80, "checkup": 60, "filling": 150, "crown": 600}


def _patient(c, pid, name, treatments, attendance):
    c.execute(
        """INSERT INTO patients(id,name,phone,age,sms_opt_in,hypertension,diabetes,
             consent_outbound,short_notice_ok,preferred_window_start,preferred_window_end,
             needed_treatments,days_waiting,attendance_history,last_offer_called_at,
             last_decline_at,last_declined_slot_type)
           VALUES (?,?,?,40,1,0,0,1,1,'08:00','19:00',?,10,?,NULL,NULL,NULL)""",
        (pid, name, f"+430{pid:06d}", json.dumps(treatments), json.dumps(attendance)))


def _slot(c, sid, ttype, when, pid, lead_days=0):
    c.execute(
        "INSERT INTO slots(id,start_dt,duration_min,type,value_eur,status,booked_patient_id,lead_days) "
        "VALUES (?,?,?,?,?,'booked',?,?)",
        (sid, when.isoformat(), 30, ttype, VAL[ttype], pid, lead_days))


@pytest.fixture(autouse=True)
def _fresh():
    import time as _t
    deadline = _t.time() + 5          # let any prior recovery threads finish (Win file lock)
    while orchestrator._in_flight and _t.time() < deadline:
        _t.sleep(0.05)
    reset_db()
    reliability.load()
    orchestrator._in_flight.clear()
    orchestrator._patient_locks.clear()
    STATE.webhook_events.clear()
    STATE.recovery = None
    prevention._client = None
    orchestrator._client = None
    yield


def _confirmation_status(sid):
    conn = connect()
    try:
        return conn.execute("SELECT confirmation_status FROM slots WHERE id=?", (sid,)).fetchone()[0]
    finally:
        conn.close()


def test_at_risk_selects_flaky_not_reliable():
    conn = connect()
    try:
        _patient(conn, 1, "Flaky", ["cleaning"], [0, 0, 0, 0, 0])
        _patient(conn, 2, "Reliable", ["cleaning"], [1, 1, 1, 1, 1])
        # flaky + long booking horizon = high risk; reliable + same-day = low risk
        _slot(conn, 1, "cleaning", datetime.now() + timedelta(days=1), 1, lead_days=60)
        _slot(conn, 2, "cleaning", datetime.now() + timedelta(days=1), 2, lead_days=0)
    finally:
        conn.close()
    risky = {sid for sid, _, _ in prevention.at_risk_slots(horizon_hours=72, threshold=0.25)}
    assert 1 in risky and 2 not in risky


def test_sweep_early_cancels_and_recovers():
    conn = connect()
    try:
        _patient(conn, 1, "Flaky", ["cleaning"], [0, 0, 0, 0, 0])
        _slot(conn, 1, "cleaning", datetime.now() + timedelta(days=1), 1, lead_days=60)
    finally:
        conn.close()
    mock = MockFonioClient(confirmation_script={1: "OUTCOME_CANCEL"})
    prevention._client = mock
    orchestrator._client = mock
    results = prevention.run_sweep(horizon_hours=72, threshold=0.25)
    assert results and results[0]["outcome"] == "cancel"
    assert _confirmation_status(1) == "cancelled"
    assert repo.recovery_attempt_for(1) is not None     # early recovery kicked off


def test_sweep_confirms_when_patient_is_coming():
    conn = connect()
    try:
        _patient(conn, 1, "Flaky", ["cleaning"], [0, 0, 0, 0, 0])
        _slot(conn, 1, "cleaning", datetime.now() + timedelta(days=1), 1, lead_days=60)
    finally:
        conn.close()
    prevention._client = MockFonioClient()      # default: everyone confirms
    results = prevention.run_sweep(horizon_hours=72, threshold=0.25)
    assert results[0]["outcome"] == "confirmed"
    assert _confirmation_status(1) == "confirmed"
