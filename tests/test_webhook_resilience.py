"""Post-call webhook receiver hardening — the Bucket-2 fixes that only the REAL
fonio client exercises (the mock writes STATE.webhook_events directly, bypassing
this HTTP endpoint). These guard the merge:

  * an UNcorrelatable webhook is ACKed with 200 and DEAD-LETTERED, never dropped
    with a 400 (a 4xx tells fonio's at-least-once retry to give up → outcome lost);
  * correlation SURVIVES A RESTART: with the in-memory STATE.pending_calls wiped,
    the webhook still finds its call via the persisted calls table.

Run it live for judges:
    cd ~/Fonio && .venv/bin/python -m pytest tests/test_webhook_resilience.py -v
"""

from __future__ import annotations

import time

import pytest

from backend import db as db_mod
from backend import repo
from backend.state import STATE


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolated DB + app TestClient, clean in-memory STATE."""
    try:
        from fastapi.testclient import TestClient
        import backend.main as main
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"webhook app stack not importable here: {e}")
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "wh.sqlite")
    db_mod.init_db()
    STATE.pending_calls.clear()
    STATE.webhook_events.clear()
    c = TestClient(main.app)        # triggers startup (init_db + reap_orphans; watchdog off in tests)
    yield c
    STATE.pending_calls.clear()
    STATE.webhook_events.clear()


def test_uncorrelated_webhook_deadletters_not_400(client):
    """No call_attempt_id, nothing in pending_calls, no matching call row → the
    payload is dead-lettered and acknowledged with 200 (so fonio stops retrying a
    payload that will never correlate), NOT rejected with a 400."""
    before = repo.count_dead_letters()
    r = client.post("/webhooks/post-call", json={
        "summary": "Patient confirmed. OUTCOME_BOOKED",
        "context": {},
    })
    assert r.status_code == 200, "must ACK (200), never 400 — a 4xx loses the outcome"
    assert r.json().get("deadletter") is True
    assert repo.count_dead_letters() == before + 1


def test_correlation_survives_restart_via_db(client):
    """Simulate a restart: STATE.pending_calls is empty, but the call row is on
    disk. A post-call webhook carrying slot_id+patient_id (no call_attempt_id) is
    correlated from the calls table and enqueued for the (re-driven) worker."""
    SID, PID, CID = 30, 4, "att-30-4-cafebabe"
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO patients(id, name, phone, age, needed_treatments, days_waiting,"
            " consent_outbound) VALUES (?,?,?,?,?,?,?)",
            (PID, "Lena Vogt", "+4399", 41, '["cleaning"]', 6, 1),
        )
        c.execute(
            "INSERT INTO slots(id, start_dt, duration_min, type, value_eur, status, lead_days)"
            " VALUES (?,?,?,?,?,?,?)",
            (SID, "2026-06-09T10:00:00", 30, "cleaning", 200, "recovering", 7),
        )
    # the call was placed (and persisted) before the 'restart'; outcome still open
    repo.log_call(fonio_call_id=CID, recovery_attempt_id=None, patient_id=PID,
                  slot_id=SID, to_number="+4399", direction="outbound",
                  outcome=None, summary=None)
    assert not STATE.pending_calls, "precondition: in-memory correlation is gone (restart)"

    r = client.post("/webhooks/post-call", json={
        "summary": "Patient confirmed they will take it. OUTCOME_BOOKED",
        "context": {"slot_id": SID, "patient_id": PID},   # NO call_attempt_id
    })
    assert r.status_code == 200
    assert r.json().get("deadletter") is not True, "should have correlated, not dead-lettered"
    assert CID in STATE.webhook_events, "outcome not enqueued under the persisted call id"
    assert repo.count_dead_letters() == 0


def test_duplicate_after_db_correlation_is_deduped(client):
    """Defense in depth: a duplicate delivery of a DB-correlated webhook is deduped."""
    SID, PID, CID = 31, 5, "att-31-5-feedface"
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO patients(id, name, phone, age, needed_treatments, days_waiting,"
            " consent_outbound) VALUES (?,?,?,?,?,?,?)",
            (PID, "Otto Berg", "+4388", 52, '["cleaning"]', 4, 1),
        )
        c.execute(
            "INSERT INTO slots(id, start_dt, duration_min, type, value_eur, status, lead_days)"
            " VALUES (?,?,?,?,?,?,?)",
            (SID, "2026-06-09T12:00:00", 30, "cleaning", 200, "recovering", 7),
        )
    repo.log_call(fonio_call_id=CID, recovery_attempt_id=None, patient_id=PID,
                  slot_id=SID, to_number="+4388", direction="outbound",
                  outcome=None, summary=None)
    payload = {"summary": "OUTCOME_BOOKED", "context": {"slot_id": SID, "patient_id": PID}}

    r1 = client.post("/webhooks/post-call", json=payload)
    assert r1.status_code == 200 and r1.json().get("duplicate") is not True
    STATE.webhook_events.pop(CID, None)            # worker consumes it
    r2 = client.post("/webhooks/post-call", json=payload)
    assert r2.status_code == 200 and r2.json().get("duplicate") is True
    assert CID not in STATE.webhook_events
