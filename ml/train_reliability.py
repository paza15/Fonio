"""Train the reliability model on Kaggle "Medical Appointment No Shows".

Per PLAN.md §5.1:
  - Dataset: joniarroba/noshowappointments → KaggleV2-May-2016.csv (~110k rows)
  - Features: Age, lead_days, SMS_received, Hipertension, Diabetes, Scholarship
  - Drop Age < 0 or Age > 110
  - Model: LightGBM(n=200, max_depth=4, class_weight="balanced")
           wrapped in CalibratedClassifierCV(method="isotonic", cv=3)
  - Report ROC-AUC (NEVER accuracy; 80% base rate). AUC > 0.9 ⇒ leakage.
  - Save bundle: { model, feature_order, auc, n, source } → ml/reliability_model.pkl

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
    df["lead_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(lower=0)
    df = df[(df["Age"] >= 0) & (df["Age"] <= 110)].copy()
    # Target: kaggle has "No-show" Yes/No. 1 = no-show, 0 = showed.
    df["no_show"] = (df["No-show"].str.lower() == "yes").astype(int)
    return df


def train():
    df = load_kaggle()
    X = df[FEATURES].astype(float).values
    y = df["no_show"].values  # predicting no-show; we serve P(showed)=1-P(no-show)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if HAVE_LGBM:
        base = LGBMClassifier(
            n_estimators=200, max_depth=4, class_weight="balanced", verbose=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        base = GradientBoostingClassifier(n_estimators=200, max_depth=4)

    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(Xtr, ytr)
    proba_no_show = model.predict_proba(Xte)[:, list(model.classes_).index(1)]
    auc = roc_auc_score(yte, proba_no_show)
    print(f"AUC (predicting no-show): {auc:.4f}  rows: {len(df):,}")
    if auc > 0.9:
        print("WARNING: AUC > 0.9 — possible label leakage. Check feature set.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_order": FEATURES, "auc": auc,
         "n": len(df), "source": "kaggle:noshowappointments KaggleV2-May-2016"},
        OUT_PATH,
    )
    (OUT_PATH.parent / "metrics.json").write_text(
        json.dumps({"auc": auc, "n": int(len(df)), "model": "LightGBM" if HAVE_LGBM else "GBM"})
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    train()
