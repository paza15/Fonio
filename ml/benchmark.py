"""Benchmark reliability models with stratified K-fold cross-validation.

Compares the production model (LightGBM) against XGBoost, Random Forest, and
AdaBoost on the Kaggle no-show data, using the SAME features as
ml/train_reliability.py (§5.1). Reports ROC-AUC mean ± std across folds.

AUC is invariant to monotonic calibration, so we benchmark the raw classifiers
(no isotonic wrapper needed) — the ranking it produces is what we care about.

Run: python -m ml.benchmark            # 5-fold on the full ~110k rows
     python -m ml.benchmark 10         # 10-fold
     python -m ml.benchmark 5 20000    # 5-fold on a 20k subsample (fast)
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml.train_reliability import FEATURES, load_kaggle

warnings.filterwarnings("ignore")

OUT_PATH = Path("ml/benchmark.json")


def build_models(pos_weight: float) -> dict:
    """pos_weight = n_neg / n_pos, used to counter the ~20% no-show imbalance."""
    models = {}

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="auc",
            tree_method="hist", n_jobs=-1,
        )
    except ImportError:
        pass

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
    return models


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    subsample = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = load_kaggle()
    if subsample:
        df = df.sample(n=min(subsample, len(df)), random_state=42)
    X = df[FEATURES].astype(float).values
    y = df["no_show"].values

    pos = y.mean()
    pos_weight = (1 - pos) / pos
    print(f"rows: {len(df):,}   no-show base rate: {pos:.1%}   "
          f"scale_pos_weight≈{pos_weight:.2f}")
    print(f"features: {FEATURES}")
    print(f"cross-validation: stratified {k}-fold (shuffle, seed=42)\n")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    models = build_models(pos_weight)

    results = []
    for name, model in models.items():
        t0 = time.time()
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        dt = time.time() - t0
        results.append({
            "model": name,
            "auc_mean": float(scores.mean()),
            "auc_std": float(scores.std()),
            "folds": [round(float(s), 4) for s in scores],
            "seconds": round(dt, 1),
        })
        print(f"{name:14s} AUC {scores.mean():.4f} ± {scores.std():.4f}   "
              f"folds={[round(float(s),3) for s in scores]}   ({dt:.1f}s)")

    results.sort(key=lambda r: r["auc_mean"], reverse=True)
    print("\n=== ranking (mean ROC-AUC, {}-fold) ===".format(k))
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['model']:14s} {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")

    OUT_PATH.write_text(json.dumps({
        "cv": f"stratified-{k}-fold",
        "n": int(len(df)),
        "base_rate": float(pos),
        "features": FEATURES,
        "results": results,
    }, indent=2))
    print(f"\nSaved {OUT_PATH}")
    best = results[0]
    print(f"Best: {best['model']} ({best['auc_mean']:.4f}). "
          f"Production model is LightGBM (see ml/train_reliability.py).")


if __name__ == "__main__":
    main()
