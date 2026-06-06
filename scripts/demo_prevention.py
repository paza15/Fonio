"""Demo: proactive no-show prevention — predict, confirm ahead, catch early.

Run: python -m scripts.demo_prevention
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ORCHESTRATOR_USE_MOCK", "true")
os.environ.setdefault("CALL_WINDOW_START", "00:00")
os.environ.setdefault("CALL_WINDOW_END", "23:59")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend import orchestrator, prevention, reliability, repo
from backend.db import init_db
from backend.fonio_client import MockFonioClient
from backend.seed import seed

HORIZON, THRESHOLD = 24 * 8, 0.30


def main():
    init_db(); seed(); reliability.load()

    risky = prevention.at_risk_slots(horizon_hours=HORIZON, threshold=THRESHOLD)
    print(f"Upcoming appointments predicted at risk (no-show ≥ {THRESHOLD:.0%}): {len(risky)}")
    for sid, p, risk in risky[:8]:
        print(f"  slot {sid:2d}  {p.name:18s}  no-show risk {risk:.0%}")
    if not risky:
        print("  (none in horizon)"); return

    # script the riskiest patient to cancel during their confirmation call
    top_sid, top_patient, _ = risky[0]
    mock = MockFonioClient(confirmation_script={top_patient.id: "OUTCOME_CANCEL"})
    prevention._client = mock
    orchestrator._client = mock

    print(f"\nRunning confirmation sweep — {len(risky)} calls "
          f"(scripted: {top_patient.name} will say they can't come)…")
    results = prevention.run_sweep(horizon_hours=HORIZON, threshold=THRESHOLD)

    print("\nOutcomes:")
    for r in results:
        print(f"  slot {r['slot_id']:2d}  {r['patient']:18s}  risk {r['risk']:.0%}  → {r['action']}")

    confirmed = sum(1 for r in results if r["outcome"] == "confirmed")
    caught = [r for r in results if r["outcome"] == "cancel"]
    print(f"\n→ {confirmed} confirmed, {len(caught)} caught early and sent to recovery "
          f"— filled (or being filled) before it ever became an empty chair.")


if __name__ == "__main__":
    main()
