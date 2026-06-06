"""Train the reliability model on Kaggle "Medical Appointment No Shows".

Per PLAN.md §5.1, extended with a reconstructed patient-history signal.

The raw Kaggle file has no "attendance history" column, but it has repeat
PatientIds (62k patients over 110k appointments). We reconstruct each patient's
*past* behaviour — leakage-safe (only appointments strictly before the current
one) and over a trailing window of 5, to exactly match the last-5
attendance_history we store per patient in the live app (backend DB).

Engineered features
  - same_day            : appointment booked for the same day (lead_days == 0)
  - prior_visits        : patient's prior appointments in the trailing-5 window
  - prior_no_show_rate  : smoothed past no-show rate (Bayesian shrink to base)
  - is_first_visit      : 1 if we've never seen this patient before

Smoothing: rate = (past_no_shows + alpha*base) / (prior_visits + alpha)
  → cold-start (first visit) falls back to the global base no-show rate.
  base + alpha are saved in the model bundle so serving uses identical values.

Split (stratified on the target):
  - train      60%  — grows the gradient-boosted trees
  - validation 20%  — early-stopping signal + isotonic calibration set
  - test       20%  — held-out, never seen → the reported success rate

We serve P(showed) = 1 - P(no-show); see backend/reliability.py.

Place the CSV at: data/kaggle/KaggleV2-May-2016.csv
Then run:        python -m ml.train_reliability
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

CSV_PATH = Path("data/kaggle/KaggleV2-May-2016.csv")
OUT_PATH = Path("ml/reliability_model.pkl")

# Features available BOTH at training time and at serving time (no train/serve
# skew): everything below maps to a field on backend.scoring.Patient / Slot.
BASELINE_FEATURES = ["Age", "lead_days", "SMS_received", "Hipertension", "Diabetes", "Scholarship"]
HISTORY_FEATURES = ["same_day", "prior_visits", "prior_no_show_rate", "is_first_visit"]
FEATURES = BASELINE_FEATURES + HISTORY_FEATURES

WINDOW = 5      # trailing attendance window — matches the app's last-5 history
ALPHA = 2.0     # Bayesian smoothing pseudo-count toward the base no-show rate


def load_kaggle() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Place Kaggle CSV at {CSV_PATH}. See README for the dataset link."
        )
    df = pd.read_csv(CSV_PATH)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["lead_days"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days.clip(lower=0)
    df = df[(df["Age"] >= 0) & (df["Age"] <= 110)].copy()
    df["no_show"] = (df["No-show"].str.lower() == "yes").astype(int)
    return df


def add_history_features(df: pd.DataFrame, base_rate: float, alpha: float = ALPHA) -> pd.DataFrame:
    """Reconstruct leakage-safe, trailing-WINDOW patient history features.

    For each appointment (ordered by patient then schedule time) we look ONLY at
    that patient's earlier appointments — never the current row's label.
    """
    df = df.sort_values(["PatientId", "ScheduledDay", "AppointmentID"]).reset_index(drop=True)
    no_show = df["no_show"].to_numpy()
    pid = df["PatientId"].to_numpy()
    n = len(df)

    prior_visits = np.zeros(n)
    prior_noshow_sum = np.zeros(n)
    seen: dict = defaultdict(lambda: deque(maxlen=WINDOW))
    for i in range(n):
        h = seen[pid[i]]
        prior_visits[i] = len(h)
        prior_noshow_sum[i] = sum(h)
        h.append(no_show[i])          # record AFTER reading → strictly past-only

    df["prior_visits"] = prior_visits
    df["is_first_visit"] = (prior_visits == 0).astype(int)
    df["prior_no_show_rate"] = (prior_noshow_sum + alpha * base_rate) / (prior_visits + alpha)
    df["same_day"] = (df["lead_days"] == 0).astype(int)
    return df


def _auc_no_show(model, X, y) -> float:
    proba = model.predict_proba(X)
    idx = list(model.classes_).index(1)
    return roc_auc_score(y, proba[:, idx])


def train():
    df = load_kaggle()
    base_rate = float(df["no_show"].mean())
    df = add_history_features(df, base_rate=base_rate, alpha=ALPHA)

    X = df[FEATURES].astype(float).values
    y = df["no_show"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print(f"rows: {len(df):,}  |  train {len(y_train):,}  val {len(y_val):,}  test {len(y_test):,}")
    print(f"no-show base rate: {base_rate:.1%}   features: {FEATURES}")
    rep = df["PatientId"].value_counts()
    print(f"repeat patients: {int((rep > 1).sum()):,}  |  appts with prior history: "
          f"{int((df['prior_visits'] > 0).sum()):,} ({(df['prior_visits'] > 0).mean():.0%})")

    best_iter = None
    if HAVE_LGBM:
        base = LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            class_weight="balanced", verbose=-1,
        )
        base.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        best_iter = int(base.best_iteration_ or base.n_estimators)
        print(f"early stopping: best_iteration = {best_iter}")
        model_name = "LightGBM"
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        base = GradientBoostingClassifier(n_estimators=200, max_depth=4)
        base.fit(X_train, y_train)
        model_name = "GradientBoosting"

    model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    model.fit(X_val, y_val)

    auc_train = _auc_no_show(model, X_train, y_train)
    auc_val = _auc_no_show(model, X_val, y_val)
    auc_test = _auc_no_show(model, X_test, y_test)
    print(f"ROC-AUC  train={auc_train:.4f}  val={auc_val:.4f}  test={auc_test:.4f}")
    if auc_test > 0.9:
        print("WARNING: test AUC > 0.9 — possible label leakage. Check the feature build.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_order": FEATURES,
            "base_rate": base_rate,      # serving must reproduce these for the
            "alpha": ALPHA,              # smoothed prior_no_show_rate feature
            "window": WINDOW,
            "auc": auc_test,
            "auc_train": auc_train,
            "auc_val": auc_val,
            "auc_test": auc_test,
            "n": len(df),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "best_iteration": best_iter,
            "source": "kaggle:noshowappointments KaggleV2-May-2016",
        },
        OUT_PATH,
    )
    (OUT_PATH.parent / "metrics.json").write_text(
        json.dumps(
            {
                "auc": auc_test, "auc_train": auc_train, "auc_val": auc_val, "auc_test": auc_test,
                "n": int(len(df)), "n_train": int(len(y_train)),
                "n_val": int(len(y_val)), "n_test": int(len(y_test)),
                "best_iteration": best_iter, "base_rate": base_rate,
                "features": FEATURES, "model": model_name,
                "source": "kaggle:noshowappointments KaggleV2-May-2016",
            },
            indent=2,
        )
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    train()
