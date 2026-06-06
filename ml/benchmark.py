"""Benchmark reliability models + feature sets with K-fold cross-validation.

Two experiments on the Kaggle no-show data (§5.1):
  1. MODELS    — LightGBM vs XGBoost vs RandomForest vs AdaBoost on the full
                 feature set (which booster is best?).
  2. FEATURES  — LightGBM on baseline-6 vs baseline + reconstructed patient
                 attendance history (does the history signal actually help?).

AUC is invariant to monotonic calibration, so we benchmark raw classifiers.

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

from ml.train_reliability import (
    ALPHA, BASELINE_FEATURES, FEATURES, add_history_features, load_kaggle,
)

warnings.filterwarnings("ignore")

OUT_PATH = Path("ml/benchmark.json")


def build_models(pos_weight: float) -> dict:
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
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
            eval_metric="auc", tree_method="hist", n_jobs=-1,
        )
    except ImportError:
        pass
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42,
    )
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
    return models


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        class_weight="balanced", verbose=-1, n_jobs=-1,
    )


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    subsample = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = load_kaggle()
    base_rate = float(df["no_show"].mean())
    df = add_history_features(df, base_rate=base_rate, alpha=ALPHA)
    if subsample:
        df = df.sample(n=min(subsample, len(df)), random_state=42)

    y = df["no_show"].values
    pos = y.mean()
    pos_weight = (1 - pos) / pos
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    print(f"rows: {len(df):,}   no-show base rate: {pos:.1%}   "
          f"cross-validation: stratified {k}-fold\n")

    # --- experiment 1: models on the full feature set ---
    print(f"[1] MODELS on full feature set ({len(FEATURES)} features)")
    X_full = df[FEATURES].astype(float).values
    model_results = []
    for name, model in build_models(pos_weight).items():
        t0 = time.time()
        s = cross_val_score(model, X_full, y, cv=cv, scoring="roc_auc", n_jobs=1)
        model_results.append({"model": name, "auc_mean": float(s.mean()), "auc_std": float(s.std())})
        print(f"    {name:14s} AUC {s.mean():.4f} ± {s.std():.4f}   ({time.time()-t0:.1f}s)")
    model_results.sort(key=lambda r: r["auc_mean"], reverse=True)

    # --- experiment 2: feature ablation (does attendance history help?) ---
    print(f"\n[2] FEATURE ABLATION (LightGBM)")
    feature_sets = {
        "baseline (6)": BASELINE_FEATURES,
        "+ attendance history (10)": FEATURES,
    }
    feat_results = []
    for label, feats in feature_sets.items():
        X = df[feats].astype(float).values
        s = cross_val_score(_lgbm(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        feat_results.append({"features": label, "n_features": len(feats),
                             "auc_mean": float(s.mean()), "auc_std": float(s.std())})
        print(f"    {label:28s} AUC {s.mean():.4f} ± {s.std():.4f}")
    lift = feat_results[1]["auc_mean"] - feat_results[0]["auc_mean"]
    print(f"    --> attendance-history lift: {lift:+.4f} AUC")

    OUT_PATH.write_text(json.dumps({
        "cv": f"stratified-{k}-fold", "n": int(len(df)), "base_rate": float(pos),
        "models": model_results, "feature_ablation": feat_results,
        "attendance_lift": float(lift),
    }, indent=2))
    print(f"\nSaved {OUT_PATH}")
    print(f"Best model: {model_results[0]['model']} ({model_results[0]['auc_mean']:.4f}); "
          f"production = LightGBM + full features.")


if __name__ == "__main__":
    main()
