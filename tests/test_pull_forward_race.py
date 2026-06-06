"""Pull-forward race-safety proof (the P2 cross-slot double-book fix).

`pull_forward_commit` does the move — fill the earlier slot + free the later one —
in ONE transaction with a compare-and-set on status='recovering'. When two
concurrent recoveries both try to pull the SAME booked patient (id 7) into two
DIFFERENT earlier slots (1 and 2), only one may win:

  * EXACTLY ONE call returns True (the other's CAS finds the slot already filled
    by the winner OR loses the BEGIN IMMEDIATE write race and rolls back).

Wait — note the two commits target DIFFERENT to_slots (1 vs 2), so the
status-CAS on the to_slot does NOT serialize them by itself. The real guard is
that the loser, on its conditional free of slot 3, finds patient 7 already gone
(the winner blanked booked_patient_id) — so the patient lands in exactly ONE of
slots 1/2, never both, and slot 3 is freed at most once.

  * patient 7 ends up booked in exactly ONE of slots 1/2 (never both);
  * slot 3 (their original booking) is freed at most once.

Run it live for judges:
    cd ~/Fonio && .venv/bin/python -m pytest tests/test_pull_forward_race.py -v
"""

from __future__ import annotations

import threading

import pytest

from backend import db as db_mod
from backend import repo


@pytest.fixture
def pf_db(tmp_path, monkeypatch):
    """Isolated SQLite DB per test (unique FONIO_DB_PATH equivalent)."""
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "pull_forward_race.sqlite")
    db_mod.init_db()
    yield


def _add_patient(pid):
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO patients(id, name, phone, age) VALUES (?,?,?,?)",
            (pid, f"P{pid}", f"+43000{pid:05d}", 40),
        )


def _add_slot(sid, status, *, booked_patient_id=None, dur=60, typ="cleaning", value=120):
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO slots(id, start_dt, duration_min, type, value_eur, status,"
            " booked_patient_id, lead_days) VALUES (?,?,?,?,?,?,?,?)",
            (sid, "2026-06-09T09:00:00", dur, typ, value, status, booked_patient_id, 7),
        )


def test_pull_forward_no_cross_slot_double_book(pf_db):
    # P7 is booked in slot 3 ('booked'); slots 1 and 2 are both 'recovering'.
    _add_patient(7)
    _add_slot(1, "recovering")                       # earlier slot A
    _add_slot(2, "recovering")                       # earlier slot C
    _add_slot(3, "booked", booked_patient_id=7)      # later slot B (their booking)

    barrier = threading.Barrier(2)
    results: dict[int, bool] = {}
    errors: dict[int, str] = {}
    lock = threading.Lock()

    def worker(to_slot: int):
        try:
            barrier.wait()  # release both threads simultaneously
            ok = repo.pull_forward_commit(3, to_slot, 7, "crown", 60, 600)
            with lock:
                results[to_slot] = ok
        except Exception as e:  # noqa: BLE001
            with lock:
                errors[to_slot] = f"{type(e).__name__}: {e}"

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"threads raised (e.g. 'database is locked'): {errors}"

    winners = [to_slot for to_slot, ok in results.items() if ok]
    assert len(winners) == 1, f"expected EXACTLY ONE winner, got {sorted(winners)}"

    with db_mod.connect() as c:
        rows = {
            r["id"]: (r["status"], r["booked_patient_id"])
            for r in c.execute(
                "SELECT id, status, booked_patient_id FROM slots WHERE id IN (1,2,3)"
            ).fetchall()
        }

    # patient 7 is booked into exactly ONE of slots 1/2, never both.
    booked_into = [sid for sid in (1, 2) if rows[sid] == ("filled", 7)]
    assert booked_into == winners, (
        f"patient 7 should be booked in exactly the winning slot {winners}, got {booked_into}"
    )
    not_won = [sid for sid in (1, 2) if sid not in winners]
    for sid in not_won:
        status, bpid = rows[sid]
        assert bpid != 7, f"patient 7 double-booked into losing slot {sid}: {rows[sid]}"
        assert status == "recovering", (
            f"losing slot {sid} must stay 'recovering' (untouched), got {status}"
        )

    # slot 3 freed at most once, and only by the winner.
    status3, bpid3 = rows[3]
    assert status3 == "cancelled", f"slot 3 should be freed once, got status {status3}"
    assert bpid3 is None, f"slot 3 should have no booked patient after the move, got {bpid3}"
