#!/usr/bin/env python3
"""XGBoost trainer for BTC predictor — trains on prediction log features."""

import json, pickle, numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DIR = Path(__file__).resolve().parent
LOG = DIR / "btc_predictions_log.json"
MODEL = DIR / "btc_xgb_model.pkl"

FEATURE_KEYS = [
    "rsi", "macd", "bb_pct", "price_vs_ma9", "price_vs_ma21", "ma_cross",
    "vol_trend", "price_change_1h", "price_change_4h",
    "ml_slope_pct", "ml_confidence",
    # New features (may not exist in old entries)
    "atr_pct", "bb_width", "macd_histogram", "macd_signal",
    "stoch_k", "stoch_d", "body_pct", "upper_wick", "lower_wick",
    "consec_green", "consec_red", "hour", "day_of_week",
    "price_change_15m", "price_change_30m", "price_change_2h",
    "range_1h", "range_4h", "vol_spike",
]

def load_data():
    log = json.load(open(LOG))
    X, y = [], []
    for e in log:
        if e.get("result") not in ("WIN", "LOSS"):
            continue
        feat = e.get("features")
        if not feat:
            continue
        row = []
        for k in FEATURE_KEYS:
            v = feat.get(k)
            row.append(float(v) if v is not None else np.nan)
        X.append(row)
        y.append(1 if e["result"] == "WIN" else 0)
    return np.array(X), np.array(y)

def train():
    X, y = load_data()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class balance: {y.sum()} wins ({y.mean()*100:.1f}%) / {len(y)-y.sum()} losses")
    print()

    # Model with tuned hyperparameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred_cv)
    auc = roc_auc_score(y, y_prob_cv)

    print("=" * 50)
    print(f"Cross-Validated Accuracy: {acc*100:.1f}%")
    print(f"Cross-Validated AUC:      {auc:.3f}")
    print("=" * 50)
    print()
    print(classification_report(y, y_pred_cv, target_names=["LOSS", "WIN"]))

    # Feature importance
    model.fit(X, y)
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    print("\nTop 15 Feature Importance:")
    print("-" * 40)
    for i in idx[:15]:
        print(f"  {FEATURE_KEYS[i]:20s}  {importances[i]:.4f}")

    # Confidence calibration (using CV probabilities)
    print("\n\nKalshi Sizing Guide (CV probabilities):")
    print("-" * 50)
    buckets = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for lo, hi in buckets:
        mask = (y_prob_cv >= lo) & (y_prob_cv < hi)
        if mask.sum() == 0:
            continue
        wr = y[mask].mean() * 100
        action = "✅ BET" if wr > 55 else "⚠️ MAYBE" if wr > 50 else "❌ SKIP"
        print(f"  {lo*100:.0f}-{hi*100:.0f}%  →  N={mask.sum():3d}  Win Rate={wr:5.1f}%  {action}")

    # Save model
    pickle.dump(model, open(MODEL, "wb"))
    print(f"\nModel saved to {MODEL}")
    print(f"Features: {len(FEATURE_KEYS)}")

if __name__ == "__main__":
    train()
