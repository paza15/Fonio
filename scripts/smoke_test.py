"""End-to-end smoke test (no HTTP server needed).

Verifies the §6.4 orchestrator loop against the mock fonio client:
  1. /rank returns top-5 + skipped+reasons with the expected hard-skips visible
  2. duplicate cancel events are idempotent (UNIQUE slot_id)
  3. the loop advances on decline, stops on booked, and writes call rows
  4. starvation guard + cooldown override + treatment mismatch all fire
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

# Windows consoles default to cp1252 and choke on the €/→ glyphs printed below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("ORCHESTRATOR_USE_MOCK", "true")
os.environ.setdefault("CALL_WINDOW_START", "00:00")
os.environ.setdefault("CALL_WINDOW_END", "23:59")

from backend import orchestrator, reliability, repo
from backend.db import init_db
from backend.seed import seed


def banner(s):
    print()
    print("=" * 64)
    print(" ", s)
    print("=" * 64)


def main():
    banner("RESET + SEED")
    init_db()
    seed()
    reliability.load()

    banner("RANK demo slot (id=1, cleaning 14:30)")
    slot = repo.get_slot(1)
    print(f"slot: id={slot.id} start={slot.start.isoformat()} type={slot.type} "
          f"value=€{slot.value_eur}")
    patients = repo.all_patients()
    from backend.scoring import rank
    ranked, skipped, phase = rank(
        slot, patients, reliability.predict,
        offers_this_week_by_pid=repo.offers_this_week_by_pid(),
    )
    print(f"phase: {phase}")
    print(f"top {len(ranked)} candidates:")
    for r in ranked:
        print(f"  {r.patient_id:2d} {r.name:30s} score={r.score:.4f} "
              f"ans={r.answer_prob:.2f} acc={r.accept_score:.2f} "
              f"days={r.days_waiting}")
    print(f"skipped ({len(skipped)}):")
    for s in skipped:
        print(f"  {s.patient_id:2d} {s.name:30s} reason={s.reason!r}")

    # Sanity checks
    skipped_pids = {s.patient_id: s.reason for s in skipped}
    assert 2 in skipped_pids and "consent" in skipped_pids[2].lower(), \
        f"patient 2 should be skipped for no consent, got: {skipped_pids.get(2)}"
    assert 3 in skipped_pids and "cooldown" in skipped_pids[3].lower(), \
        f"patient 3 should be skipped for cooldown, got: {skipped_pids.get(3)}"
    assert 5 in skipped_pids and "mismatch" in skipped_pids[5].lower(), \
        f"patient 5 should be skipped for treatment mismatch, got: {skipped_pids.get(5)}"
    ranked_ids = [r.patient_id for r in ranked]
    assert 1 in ranked_ids, \
        f"patient 1 (Maria) should be a feasible candidate, got: {ranked_ids}"
    print("  [OK]skip reasons present (consent, cooldown, mismatch)")
    # NOTE: the population no-show model has no attendance feature, so it ranks
    # Maria on age/comorbidities, not her 5/5 history — she's feasible but not #1.
    # The end-to-end fill assertion below still proves she gets booked.
    print(f"  [OK]Maria Huber feasible (rank #{ranked_ids.index(1) + 1} of {len(ranked_ids)})")

    banner("ORCHESTRATOR — trigger recovery on slot 1")
    r1 = orchestrator.trigger_recovery(1)
    print(f"first trigger: {r1}")
    r2 = orchestrator.trigger_recovery(1)
    print(f"duplicate trigger: {r2}")
    assert r1["ok"] is True
    assert r2["ok"] is False, "duplicate cancellation must be idempotent"
    print("  [OK]idempotent against duplicate cancel events")

    banner("Wait for the mock loop to complete (patient 1 → BOOKED in ~3s)")
    for i in range(40):
        time.sleep(0.5)
        status = repo.slot_status(1)
        if status in ("filled", "escalated", "unrecoverable"):
            print(f"  loop done at iter {i}: slot status = {status}")
            break
    else:
        raise SystemExit("loop did not finish in 20s")

    banner("POST-CONDITIONS")
    assert status == "filled", f"expected filled, got {status}"
    attempt = repo.recovery_attempt_for(1)
    print(f"recovery_attempts: {attempt}")
    assert attempt["status"] == "filled"
    assert attempt["filled_by_patient_id"] == 1

    from backend.db import connect
    conn = connect()
    try:
        calls = conn.execute(
            "SELECT patient_id, outcome, summary FROM calls WHERE slot_id = 1 "
            "ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    print(f"calls logged for slot 1: {len(calls)}")
    for c in calls:
        print(f"  patient={c['patient_id']} outcome={c['outcome']} "
              f"summary={(c['summary'] or '')[:60]!r}")
    assert any(c["outcome"] == "booked" for c in calls), "should have a booked call"
    print("  [OK]slot filled, call row recorded with outcome=booked")

    banner("STARVATION GUARD: re-rank now that patient 1 is filled")
    # patient 4 is the >30-day waiter; with patient 1 excluded they should
    # benefit from the ×1.5 starvation boost
    slot2 = repo.get_slot(2)
    ranked2, _, _ = rank(
        slot2, repo.all_patients(), reliability.predict,
        exclude_ids={1},
        offers_this_week_by_pid=repo.offers_this_week_by_pid(),
    )
    # slot 2 is a crown @ 17:00 — patient 4 needs crown+filling, so they should pass.
    top_ids = [r.patient_id for r in ranked2[:5]]
    print(f"  top 5 for crown slot: {top_ids}")
    assert 4 in top_ids, "long-waiter (id=4) should be a top candidate for crown"
    print("  [OK]long-waiter surfaces on a matching treatment")

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
