"""Train the reliability model on Kaggle "Medical Appointment No Shows".

Per PLAN.md §5.1:
  - Dataset: joniarroba/noshowappointments → KaggleV2-May-2016.csv (~110k rows)
  - Features: Age, lead_days, SMS_received, Hipertension, Diabetes, Scholarship
  - Drop Age < 0 or Age > 110
  - Model: LightGBM(max_depth=4, class_weight="balanced") + isotonic calibration
  - Report ROC-AUC (NEVER accuracy; ~80% base rate). AUC > 0.9 ⇒ leakage.

Split (stratified on the target):
  - train      60%  — grows the gradient-boosted trees
  - validation 20%  — early-stopping signal + isotonic calibration set
  - test       20%  — held-out, never seen in training → the reported success rate

We serve P(showed) = 1 - P(no-show); see backend/reliability.py.

Place the CSV at: data/kaggle/KaggleV2-May-2016.csv
Then run:        python -m ml.train_reliability
"""

from __future__ import annotations

import json
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
FEATURES = ["Age", "lead_days", "SMS_received", "Hipertension", "Diabetes", "Scholarship"]


def load_kaggle() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Place Kaggle CSV at {CSV_PATH}. See README for the dataset link."
        )
    df = pd.read_csv(CSV_PATH)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    # AppointmentDay has no time component → normalize both to dates for lead_days.
    df["lead_days"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days.clip(lower=0)
    df = df[(df["Age"] >= 0) & (df["Age"] <= 110)].copy()
    # Target: kaggle has "No-show" Yes/No. 1 = no-show, 0 = showed.
    df["no_show"] = (df["No-show"].str.lower() == "yes").astype(int)
    return df


def _auc_no_show(model, X, y) -> float:
    """ROC-AUC for predicting no-show (class 1)."""
    proba = model.predict_proba(X)
    idx = list(model.classes_).index(1)
    return roc_auc_score(y, proba[:, idx])


def train():
    df = load_kaggle()
    X = df[FEATURES].astype(float).values
    y = df["no_show"].values  # 1 = no-show

    # 60 / 20 / 20 stratified split. First carve off 40% (val+test), then halve it.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print(
        f"rows: {len(df):,}  |  train {len(y_train):,}  "
        f"val {len(y_val):,}  test {len(y_test):,}"
    )
    print(f"no-show base rate: {y.mean():.1%}")

    best_iter = None
    if HAVE_LGBM:
        base = LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            class_weight="balanced", verbose=-1,
        )
        # Validation set drives early stopping (picks the tree count).
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

    # Isotonic calibration on the validation set (base is already fitted → prefit).
    model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    model.fit(X_val, y_val)

    auc_train = _auc_no_show(model, X_train, y_train)
    auc_val = _auc_no_show(model, X_val, y_val)
    auc_test = _auc_no_show(model, X_test, y_test)
    print(f"ROC-AUC  train={auc_train:.4f}  val={auc_val:.4f}  test={auc_test:.4f}")
    if auc_test > 0.9:
        print("WARNING: test AUC > 0.9 — possible label leakage. Check the feature set.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_order": FEATURES,
            "auc": auc_test,          # headline = held-out test AUC
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
                "auc": auc_test,
                "auc_train": auc_train,
                "auc_val": auc_val,
                "auc_test": auc_test,
                "n": int(len(df)),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "best_iteration": best_iter,
                "model": model_name,
                "source": "kaggle:noshowappointments KaggleV2-May-2016",
            },
            indent=2,
        )
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    train()
