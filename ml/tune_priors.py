"""Tune the learned-signal prior strength (scoring.ANSWER/ACCEPT_PRIOR_STRENGTH).

Those constants set how many real call outcomes it takes before a patient's
observed answer/accept rate outweighs the prior. There is no public call-log
data — but the per-patient *shrinkage* problem is identical to predicting a
patient's next no-show from their past ones, and Kaggle gives us 110k
appointments across 62k repeat patients.

So we tune k there, leakage-safe: for each appointment, predict its outcome from
the patient's trailing-WINDOW PRIOR outcomes, shrunk toward the global base rate

    p = (past_no_shows + k * base) / (past_visits + k)

This is a proper "predict the next event" evaluation (strictly past inputs).
Pick k by predictive log-loss. The result transfers to call answer/accept, which
is the same repeated-binary-rate structure.

Run: python -m ml.tune_priors
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ml.train_reliability import WINDOW, load_kaggle

warnings.filterwarnings("ignore")
EPS = 1e-6
OUT = Path("ml/prior_tuning.json")
GRID = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e12]


def trailing_counts(df):
    """Leakage-safe trailing-WINDOW prior counts per appointment (matches the
    app's last-5 history and ml/train_reliability.add_history_features)."""
    df = df.sort_values(["PatientId", "ScheduledDay", "AppointmentID"]).reset_index(drop=True)
    no_show = df["no_show"].to_numpy()
    pid = df["PatientId"].to_numpy()
    n = len(df)
    visits = np.zeros(n)
    noshow_sum = np.zeros(n)
    seen = defaultdict(lambda: deque(maxlen=WINDOW))
    for i in range(n):
        h = seen[pid[i]]
        visits[i] = len(h)
        noshow_sum[i] = sum(h)
        h.append(no_show[i])
    return visits, noshow_sum, no_show


def main():
    df = load_kaggle()
    base = float(df["no_show"].mean())
    visits, nss, y = trailing_counts(df)
    mask = visits >= 1                      # rows where prior history (and k) matter
    v, s, yy = visits[mask], nss[mask], y[mask]
    print(f"base no-show rate: {base:.3f}   eval rows (≥1 prior visit): {int(mask.sum()):,}\n")
    print(f"{'k':>12} {'logloss':>10} {'brier':>9} {'auc':>8}")

    results = []
    for k in GRID:
        p = np.clip((s + k * base) / (v + k), EPS, 1 - EPS)
        ll = log_loss(yy, p)
        br = brier_score_loss(yy, p)
        auc = roc_auc_score(yy, p)
        if k == 0:
            label = "0 (empirical)"
        elif k > 1e8:
            label = "inf (base)"
        else:
            label = f"{k:g}"
        results.append({"k": (None if k > 1e8 else float(k)),
                        "logloss": ll, "brier": br, "auc": auc})
        print(f"{label:>12} {ll:10.5f} {br:9.5f} {auc:8.4f}")

    finite = [r for r in results if r["k"] is not None and r["k"] > 0]
    best = min(finite, key=lambda r: r["logloss"])
    base_only = next(r for r in results if r["k"] is None)
    empirical = next(r for r in results if r["k"] == 0.0)
    print(f"\nBest k by predictive log-loss: {best['k']:g}  (logloss {best['logloss']:.5f})")
    print(f"  vs base-only (k→inf):  {base_only['logloss']:.5f}   "
          f"({base_only['logloss'] - best['logloss']:+.5f})")
    print(f"  vs pure-empirical (k=0): {empirical['logloss']:.5f}   "
          f"({empirical['logloss'] - best['logloss']:+.5f})")

    OUT.write_text(json.dumps(
        {"base": base, "eval_rows": int(mask.sum()), "window": WINDOW,
         "grid": results, "best_k": best["k"]}, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
