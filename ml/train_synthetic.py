"""Synthetic fallback trainer.

Used when the Kaggle CSV isn't available — generates a plausible no-show
dataset with the same feature schema, trains the same pipeline, and writes
the same bundle. README must label this clearly: the AUC here is meaningless
for real performance, the artifact only keeps the `/rank` path consistent.

Run: `python -m ml.train_synthetic`
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

OUT_PATH = Path("ml/reliability_model.pkl")
FEATURES = ["Age", "lead_days", "SMS_received", "Hipertension", "Diabetes", "Scholarship"]


def synth(n: int = 20_000, seed: int = 7):
    rng = np.random.default_rng(seed)
    age = rng.integers(5, 90, size=n)
    lead = rng.integers(0, 90, size=n)
    sms = rng.integers(0, 2, size=n)
    hyp = (rng.random(n) < 0.18).astype(int)
    dia = (rng.random(n) < 0.10).astype(int)
    sch = (rng.random(n) < 0.07).astype(int)
    logit = (
        -1.4
        + 0.018 * (lead - 5)
        - 0.6 * sms
        - 0.005 * age
        + 0.15 * hyp
        + 0.10 * dia
        + 0.20 * sch
        + rng.normal(0, 0.4, size=n)
    )
    no_show = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    X = np.stack([age, lead, sms, hyp, dia, sch], axis=1).astype(float)
    return X, no_show


def train():
    X, y = synth()
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
    print(f"[synthetic] AUC: {auc:.4f}  (NOT REPRESENTATIVE — synthetic data)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_order": FEATURES, "auc": auc,
         "n": len(X), "source": "synthetic-fallback"},
        OUT_PATH,
    )
    (OUT_PATH.parent / "metrics.json").write_text(
        json.dumps({"auc": auc, "n": int(len(X)),
                    "model": "LightGBM" if HAVE_LGBM else "GBM",
                    "source": "synthetic-fallback"})
    )
    print(f"Saved {OUT_PATH} (synthetic fallback)")


if __name__ == "__main__":
    train()
