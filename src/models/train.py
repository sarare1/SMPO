"""Train and evaluate all predictive models.

WHAT THIS FILE DOES (plain English)
-----------------------------------
Step 3 of the pipeline. It teaches the computer to recognise the warning signs
of failure by showing it historical examples, then checks how well it learned
on examples it has never seen ("test data").

AI4I 2020 (machining line):
    * one model that predicts whether a production cycle will fail, and
    * one model per failure type (tool wear, heat, power, overstrain).
Azure PdM (machine fleet):
    * for each of the 4 components, a model predicting failure within 24 hours
      and another predicting failure within 7 days (8 models), and
    * an "anomaly detector" that flags unusual sensor behaviour.

The trained models are saved in models/, and their accuracy scores in
reports/metrics.json (shown on the dashboard's "Model performance" page).

Glossary of the scores:
    precision - of the alarms raised, how many were real failures.
    recall    - of the real failures, how many were caught.
    PR-AUC / ROC-AUC - overall ranking quality, 1.0 = perfect.
    Brier     - how trustworthy the probabilities are, 0 = perfect.

Usage:  python -m src.models.train
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import json  # saves the accuracy scores as a readable text file

import joblib           # saves trained models to disk and loads them back
import lightgbm as lgb  # LightGBM: the prediction algorithm (many small decision trees)
import numpy as np      # fast maths on lists of numbers
import pandas as pd     # tables of data
from sklearn.ensemble import IsolationForest  # finds unusual data points (anomalies)
from sklearn.metrics import (                 # formulas that score how good a model is
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # data splitting

from src import config  # project settings: folders, split date, random seed
from src.data.features import (  # column lists defined during feature engineering
    AI4I_FEATURES,
    AI4I_MODES,
    ANOMALY_FEATURES,
    AZURE_FEATURES,
    AZURE_LABELS,
    COMPONENTS,
)

# Settings for every LightGBM model: 400 small decision trees, learning gradually,
# each tree seeing a random 80% of rows and columns (makes the model more robust).
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
    """Create a new, untrained prediction model with the standard settings."""
    # No class re-weighting: the scheduler multiplies probabilities by dollar
    # costs, so they must stay calibrated. Class imbalance is handled by
    # tuning the alert threshold instead.
    return lgb.LGBMClassifier(**CLF_PARAMS)


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Find the probability cut-off (e.g. 0.47) above which we raise an alarm.

    Tries 91 cut-offs between 0.05 and 0.95 and keeps the one with the best
    balance of catching failures (recall) and avoiding false alarms (precision).
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _binary_metrics(y_true, proba, threshold) -> dict:
    """Score a model: compare its predictions on test data with what really happened."""
    pred = proba >= threshold  # alarm (True) or no alarm (False) for every row
    return {
        "brier": round(float(brier_score_loss(y_true, proba)), 5),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "threshold": round(threshold, 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),  # counts of right/wrong alarms
        "positives_in_test": int(np.sum(y_true)),                     # real failures in the test data
    }


def train_ai4i() -> dict:
    """Train and score the machining-line models (overall failure + each failure type)."""
    df = pd.read_parquet(config.DATA_PROCESSED / "ai4i_features.parquet")
    X, y = df[AI4I_FEATURES], df["machine_failure"]  # X = inputs, y = the answer to learn
    # Keep 20% of cycles aside as unseen test data, with the same share of failures in both parts.
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    # Decision threshold chosen on out-of-fold train predictions (no test leakage)
    # (The training data is split 5 ways; each part is predicted by a model trained
    # on the other 4, so the cut-off is chosen without ever looking at test data.)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(_classifier(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = _best_f1_threshold(y_tr.values, oof)

    # Train the overall failure model on all training data and score it on the test data.
    failure_model = _classifier().fit(X_tr, y_tr)
    metrics = {"machine_failure": _binary_metrics(y_te.values, failure_model.predict_proba(X_te)[:, 1], threshold)}

    # One model per failure type, trained on the same rows.
    mode_models = {}
    for mode in AI4I_MODES:
        ym_tr, ym_te = df.loc[idx_tr, mode], df.loc[idx_te, mode]
        model = _classifier().fit(X_tr, ym_tr)
        mode_models[mode] = model
        metrics[mode] = _binary_metrics(ym_te.values, model.predict_proba(X_te)[:, 1], 0.5)

    # Save all machining-line models and the alarm cut-off in one file.
    joblib.dump(
        {"failure": failure_model, "modes": mode_models, "threshold": threshold, "features": AI4I_FEATURES},
        config.MODELS_DIR / "ai4i_models.joblib",
    )
    # Store the held-out rows so the dashboard only demos unseen data
    df.loc[idx_te].to_parquet(config.DATA_PROCESSED / "ai4i_test.parquet")
    return metrics


def train_azure() -> dict:
    """Train and score the fleet models (24h and 7-day failure per component, anomaly detector)."""
    df = pd.read_parquet(config.DATA_PROCESSED / "azure_features.parquet")
    # Split by date: learn from Jan-Aug 2015, test on Sep 2015 onwards (the "future").
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
    # ("Unsupervised" = it learns what normal looks like, without failure examples.)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=config.RANDOM_STATE)
    iso.fit(train[ANOMALY_FEATURES].fillna(0).sample(60_000, random_state=config.RANDOM_STATE))
    # Higher score = more unusual.
    train_scores = -iso.score_samples(train[ANOMALY_FEATURES].fillna(0))
    test_scores = -iso.score_samples(test[ANOMALY_FEATURES].fillna(0))
    p99 = float(np.percentile(train_scores, 99))  # "unusual" = stranger than 99% of normal operation
    # Check: are readings just before a failure flagged more often than readings in general?
    any_fail = test[AZURE_LABELS["24h"]].max(axis=1).values.astype(bool)
    metrics["anomaly"] = {
        "p99_threshold": round(p99, 4),
        "flag_rate_test": round(float((test_scores > p99).mean()), 4),
        "flag_rate_24h_before_failure": round(float((test_scores[any_fail] > p99).mean()), 4),
    }

    # Save all fleet models in one file, with the reference values used to scale anomaly scores.
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
    """Train everything, save the scores to reports/metrics.json and print a summary."""
    metrics = {"ai4i": train_ai4i(), "azure_pdm": train_azure()}
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for ds, block in metrics.items():
        print(f"\n== {ds} ==")
        for name, m in block.items():
            short = {k: v for k, v in m.items() if k != "confusion_matrix"}
            print(f"{name:>18}: {short}")


# Runs only when this file is started directly (python -m src.models.train).
if __name__ == "__main__":
    main()
