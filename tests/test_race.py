"""Race-safety + idempotency proof for the recovery loop (the 25% claim).

This is the committed, reproducible version of the adversarial checks that
guard the sequential model:

  * try_fill_slot is an ATOMIC compare-and-set: when N writers race for the
    SAME slot, EXACTLY ONE wins (the others no-op — no double-book).
  * a late/duplicate writer cannot overwrite an already-filled slot.
  * duplicate CANCELLATION events are rejected by UNIQUE(slot_id).
  * duplicate POST-CALL webhooks are deduped by processed_events.
  * end-to-end: the same post-call payload delivered twice fills the slot once
    (defense-in-depth: webhook dedup + the try_fill_slot CAS).

Run it live for judges:
    cd ~/Fonio && .venv/bin/python -m pytest tests/test_race.py -v
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from backend import db as db_mod
from backend import repo
from backend.state import STATE


@pytest.fixture
def race_db(tmp_path, monkeypatch):
    """Isolated SQLite DB per test + clean in-memory STATE."""
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "race.sqlite")
    db_mod.init_db()
    STATE.pending_calls.clear()
    STATE.webhook_events.clear()
    yield
    STATE.pending_calls.clear()
    STATE.webhook_events.clear()


def _add_slot(sid, status="recovering", value_eur=120):
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO slots(id, start_dt, duration_min, type, value_eur, status, lead_days)"
            " VALUES (?,?,?,?,?,?,?)",
            (sid, "2026-06-09T09:00:00", 30, "cleaning", value_eur, status, 7),
        )


def _add_patient(pid):
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO patients(id, name, phone, age) VALUES (?,?,?,?)",
            (pid, f"P{pid}", f"+43000{pid:05d}", 40),
        )


# --- the core race: N writers, one slot, exactly one winner ---

def test_atomic_commit_exactly_one_winner(race_db):
    SLOT_ID, N = 1, 20
    _add_slot(SLOT_ID)
    for pid in range(1, N + 1):
        _add_patient(pid)

    barrier = threading.Barrier(N)
    results: dict[int, bool] = {}
    errors: dict[int, str] = {}
    lock = threading.Lock()

    def worker(pid: int):
        try:
            barrier.wait()  # release all threads simultaneously
            ok = repo.try_fill_slot(SLOT_ID, pid)
            with lock:
                results[pid] = ok
        except Exception as e:  # noqa: BLE001
            with lock:
                errors[pid] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=worker, args=(pid,)) for pid in range(1, N + 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"threads raised (e.g. 'database is locked'): {errors}"
    winners = [pid for pid, ok in results.items() if ok]
    assert len(winners) == 1, f"expected EXACTLY ONE winner, got {sorted(winners)}"

    with db_mod.connect() as c:
        row = c.execute(
            "SELECT status, booked_patient_id FROM slots WHERE id=?", (SLOT_ID,)
        ).fetchone()
    assert row["status"] == "filled"
    assert row["booked_patient_id"] == winners[0], "committed booking != winning writer"


def test_late_writer_cannot_overwrite(race_db):
    SLOT_ID = 1
    _add_slot(SLOT_ID)
    _add_patient(5)
    _add_patient(999)

    assert repo.try_fill_slot(SLOT_ID, 5) is True
    assert repo.try_fill_slot(SLOT_ID, 999) is False, "late writer must be rejected"

    with db_mod.connect() as c:
        row = c.execute(
            "SELECT status, booked_patient_id FROM slots WHERE id=?", (SLOT_ID,)
        ).fetchone()
    assert row["status"] == "filled" and row["booked_patient_id"] == 5


# --- duplicate cancellation event ---

def test_duplicate_cancellation_unique_guard(race_db):
    SID = 10
    _add_slot(SID)

    first = repo.create_recovery_attempt(SID)
    assert isinstance(first, int) and first > 0

    with pytest.raises(sqlite3.IntegrityError):
        repo.create_recovery_attempt(SID)  # UNIQUE(slot_id) rejects the duplicate

    with db_mod.connect() as c:
        n = c.execute(
            "SELECT COUNT(*) AS n FROM recovery_attempts WHERE slot_id=?", (SID,)
        ).fetchone()["n"]
    assert n == 1


# --- duplicate post-call webhook: dedup primitive ---

def test_duplicate_webhook_dedup_primitive(race_db):
    ek = "postcall:att-1-2-deadbeef"
    assert repo.mark_event_processed(ek, "post_call") is True
    assert repo.mark_event_processed(ek, "post_call") is False  # deduped

    with db_mod.connect() as c:
        n = c.execute(
            "SELECT COUNT(*) AS n FROM processed_events WHERE event_key=?", (ek,)
        ).fetchone()["n"]
    assert n == 1


# --- duplicate post-call webhook: end-to-end (webhook dedup + CAS) ---

def test_duplicate_post_call_webhook_end_to_end(race_db):
    try:
        from fastapi.testclient import TestClient
        import backend.main as main
    except Exception as e:  # noqa: BLE001 — needs the full app stack
        pytest.skip(f"webhook app stack not importable here: {e}")

    SID, PID = 20, 3
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO patients(id, name, phone, age, needed_treatments, days_waiting,"
            " consent_outbound) VALUES (?,?,?,?,?,?,?)",
            (PID, "Eva Klein", "+433", 33, '["cleaning"]', 5, 1),
        )
        c.execute(
            "INSERT INTO slots(id, start_dt, duration_min, type, value_eur, status, lead_days)"
            " VALUES (?,?,?,?,?,?,?)",
            (SID, "2026-06-09T11:00:00", 30, "cleaning", 200, "recovering", 7),
        )
        c.execute(
            "INSERT INTO recovery_attempts(slot_id, created_at, status)"
            " VALUES (?,?,?)",
            (SID, "2026-06-06T00:00:00", "in_progress"),
        )

    cid = f"att-{SID}-{PID}-deadbeef"
    STATE.pending_calls[cid] = {"slot_id": SID, "patient_id": PID, "to_number": "+433"}
    payload = {
        "summary": "Patient confirmed they will take the slot. OUTCOME_BOOKED",
        "context": {"slot_id": SID, "patient_id": PID, "call_attempt_id": cid},
    }
    client = TestClient(main.app)

    # First delivery: correlates, enqueues, NOT a duplicate.
    r1 = client.post("/webhooks/post-call", json=payload)
    assert r1.status_code == 200
    assert r1.json().get("duplicate") is not True
    assert cid in STATE.webhook_events

    # Orchestrator-style consume → atomic fill.
    assert STATE.webhook_events.pop(cid, None) is not None
    assert repo.try_fill_slot(SID, PID) is True
    repo.finish_recovery(SID, "filled", PID)
    assert repo.slot_status(SID) == "filled"

    # Second (duplicate) delivery: deduped at the webhook, not re-enqueued.
    r2 = client.post("/webhooks/post-call", json=payload)
    assert r2.status_code == 200
    assert r2.json().get("duplicate") is True
    assert cid not in STATE.webhook_events

    # Defense-in-depth: even a re-consume can't double-book.
    assert repo.try_fill_slot(SID, PID) is False
