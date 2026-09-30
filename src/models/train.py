"""Train and evaluate all predictive models.

AI4I 2020:  machine-failure classifier + one classifier per failure mode.
Azure PdM:  24h and 7-day failure classifiers per component,
            Isolation Forest anomaly detector.

Usage:  python -m src.models.train
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src import config
from src.data.features import (
    AI4I_FEATURES,
    AI4I_MODES,
    ANOMALY_FEATURES,
    AZURE_FEATURES,
    AZURE_LABELS,
    COMPONENTS,
)

CLF_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=config.RANDOM_STATE,
    verbose=-1,
)


def _classifier() -> lgb.LGBMClassifier:
    # No class re-weighting: the scheduler multiplies probabilities by dollar
    # costs, so they must stay calibrated. Class imbalance is handled by
    # tuning the alert threshold instead.
    return lgb.LGBMClassifier(**CLF_PARAMS)


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _binary_metrics(y_true, proba, threshold) -> dict:
    pred = proba >= threshold
    return {
        "brier": round(float(brier_score_loss(y_true, proba)), 5),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "threshold": round(threshold, 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "positives_in_test": int(np.sum(y_true)),
    }


def train_ai4i() -> dict:
    df = pd.read_parquet(config.DATA_PROCESSED / "ai4i_features.parquet")
    X, y = df[AI4I_FEATURES], df["machine_failure"]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    # Decision threshold chosen on out-of-fold train predictions (no test leakage)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(_classifier(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = _best_f1_threshold(y_tr.values, oof)

    failure_model = _classifier().fit(X_tr, y_tr)
    metrics = {"machine_failure": _binary_metrics(y_te.values, failure_model.predict_proba(X_te)[:, 1], threshold)}

    mode_models = {}
    for mode in AI4I_MODES:
        ym_tr, ym_te = df.loc[idx_tr, mode], df.loc[idx_te, mode]
        model = _classifier().fit(X_tr, ym_tr)
        mode_models[mode] = model
        metrics[mode] = _binary_metrics(ym_te.values, model.predict_proba(X_te)[:, 1], 0.5)

    joblib.dump(
        {"failure": failure_model, "modes": mode_models, "threshold": threshold, "features": AI4I_FEATURES},
        config.MODELS_DIR / "ai4i_models.joblib",
    )
    # Store the held-out rows so the dashboard only demos unseen data
    df.loc[idx_te].to_parquet(config.DATA_PROCESSED / "ai4i_test.parquet")
    return metrics


def train_azure() -> dict:
    df = pd.read_parquet(config.DATA_PROCESSED / "azure_features.parquet")
    train = df[df["datetime"] < config.AZURE_SPLIT_DATE]
    test = df[df["datetime"] >= config.AZURE_SPLIT_DATE]
    X_tr, X_te = train[AZURE_FEATURES], test[AZURE_FEATURES]

    # One classifier per component and horizon. The 7-day horizon feeds the
    # weekly maintenance scheduler; the 24h horizon drives urgent alerts.
    metrics, comp_models = {}, {}
    data_end = df["datetime"].max()
    for tag, hours in config.HORIZONS_H.items():
        comp_models[tag] = {}
        # Rows too close to the end of the data have unobservable labels
        te = test[test["datetime"] <= data_end - pd.Timedelta(hours=hours)]
        for comp, label in zip(COMPONENTS, AZURE_LABELS[tag]):
            model = _classifier().fit(X_tr, train[label])
            comp_models[tag][comp] = model
            proba = model.predict_proba(te[AZURE_FEATURES])[:, 1]
            metrics[label] = _binary_metrics(te[label].values, proba, 0.5)

    # Unsupervised anomaly detector for patterns the supervised models never saw
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=config.RANDOM_STATE)
    iso.fit(train[ANOMALY_FEATURES].fillna(0).sample(60_000, random_state=config.RANDOM_STATE))
    train_scores = -iso.score_samples(train[ANOMALY_FEATURES].fillna(0))
    test_scores = -iso.score_samples(test[ANOMALY_FEATURES].fillna(0))
    p99 = float(np.percentile(train_scores, 99))
    any_fail = test[AZURE_LABELS["24h"]].max(axis=1).values.astype(bool)
    metrics["anomaly"] = {
        "p99_threshold": round(p99, 4),
        "flag_rate_test": round(float((test_scores > p99).mean()), 4),
        "flag_rate_24h_before_failure": round(float((test_scores[any_fail] > p99).mean()), 4),
    }

    joblib.dump(
        {
            "components": comp_models,
            "anomaly": iso,
            "anomaly_ref": np.percentile(train_scores, [50, 99]).tolist(),
            "features": AZURE_FEATURES,
            "anomaly_features": ANOMALY_FEATURES,
        },
        config.MODELS_DIR / "azure_models.joblib",
    )
    return metrics


def main() -> None:
    metrics = {"ai4i": train_ai4i(), "azure_pdm": train_azure()}
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for ds, block in metrics.items():
        print(f"\n== {ds} ==")
        for name, m in block.items():
            short = {k: v for k, v in m.items() if k != "confusion_matrix"}
            print(f"{name:>18}: {short}")


if __name__ == "__main__":
    main()
