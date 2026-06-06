"""Demo: the engine learns each patient's responsiveness from its own call log.

We rank a slot cold (priors only), then feed in realistic past-call outcomes for
the two front-runners and re-rank. Their P(answer)/P(accept) get shrunk toward
what actually happened on the phone — so a chronic decliner and a never-answers
patient drop, and a quieter, reliable candidate rises.

Run: python -m scripts.demo_learning
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

os.environ.setdefault("ORCHESTRATOR_USE_MOCK", "true")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend import reliability, repo
from backend.db import init_db
from backend.scoring import rank
from backend.seed import seed


def log_history(pid: int, outcomes: list[str], days_ago: int = 30) -> None:
    """Insert past *completed* offer calls so they count toward learned stats
    but not the 7-day weekly contact cap (hence days_ago=30)."""
    for i, oc in enumerate(outcomes):
        ts = (datetime.now() - timedelta(days=days_ago, minutes=i)).isoformat()
        repo.log_call(
            fonio_call_id=f"hist_{pid}_{i}_{oc}", recovery_attempt_id=None,
            patient_id=pid, slot_id=1, direction="outbound",
            outcome=oc, summary="(historical)", started_at=ts, ended_at=ts,
        )


def show(title: str, ranked) -> None:
    print(f"\n{title}")
    print(f"  {'#':>2} {'patient':16} {'score':>7} {'P(ans)':>7} {'P(acc)':>7}  call record")
    for i, r in enumerate(ranked, 1):
        offers, answered, accepted = r.call_history
        rec = f"{answered}/{offers} answered, {accepted} booked" if offers else "— (no history)"
        print(f"  {i:>2} {r.name:16} {r.score:7.4f} {r.answer_prob:7.3f} "
              f"{r.accept_score:7.3f}  {rec}")


def main():
    init_db(); seed(); reliability.load()
    slot = repo.get_slot(1)

    cold, _, _ = rank(
        slot, repo.all_patients(), reliability.predict,
        offers_this_week_by_pid=repo.offers_this_week_by_pid(),
    )
    show("BEFORE — cold start, priors only (model P(show) + heuristic accept):", cold)

    leader = cold[0].patient_id
    runner_up = cold[1].patient_id
    print(f"\n  → logging call history:")
    print(f"     • {cold[0].name} (#1): answered 5 offers, declined ALL 5")
    print(f"     • {cold[1].name} (#2): 4 calls, never picked up (voicemail)")
    log_history(leader, ["declined"] * 5)
    log_history(runner_up, ["voicemail"] * 4)

    warm, _, _ = rank(
        slot, repo.all_patients(), reliability.predict,
        offers_this_week_by_pid=repo.offers_this_week_by_pid(),
        call_stats_by_pid=repo.call_stats_by_pid(),
    )
    show("AFTER — same slot, now shrinking priors toward the real call record:", warm)

    moved = {r.patient_id: i for i, r in enumerate(warm)}
    print("\n  what changed:")
    for r in cold[:2]:
        before = next(i for i, x in enumerate(cold) if x.patient_id == r.patient_id)
        after = moved.get(r.patient_id)
        where = f"#{after + 1}" if after is not None else "dropped out of top-5"
        print(f"     • {r.name}: #{before + 1} → {where}")
    print(f"     • new #1: {warm[0].name}")
    print("\n  The heuristic was optimistic; the call log corrected it — no model "
          "retrain, just the engine learning from its own outcomes.")


if __name__ == "__main__":
    main()
