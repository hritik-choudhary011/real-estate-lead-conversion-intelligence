"""Conversion-likelihood models: baseline (dummy + logistic regression) and Random Forest.

Leakage controls
----------------
* ``lead_status`` is written after the outcome is known -> excluded.
* ``lead_id``, ``converted`` and ``campaign_name`` are excluded (see config.EXCLUDED_FROM_MODEL).
* Train/test split is stratified; cross-validated scores use training data only.
* Scored probabilities shown to users are OUT-OF-FOLD (each lead is scored by a
  model that never saw that lead), so the dashboard does not look better than it is.
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "Baseline (majority class)": Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=config.RANDOM_STATE)),
        ]),
    }


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def _metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


def lift_table(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: decile 1 = highest predicted probability."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    t = d.groupby("decile", observed=True).agg(leads=("y", "size"), conversions=("y", "sum"),
                                              conv_rate=("y", "mean"), avg_prob=("p", "mean")).reset_index()
    t["lift"] = t["conv_rate"] / d["y"].mean()
    t["cum_capture"] = t["conversions"].cumsum() / d["y"].sum()
    return t


def train_and_evaluate(df: pd.DataFrame, save: bool = True) -> dict[str, Any]:
    """Fit all models, evaluate on the hold-out set, return a results bundle."""
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results: dict[str, Any] = {"models": {}, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                               "positive_rate_train": round(float(y_train.mean()), 4),
                               "positive_rate_test": round(float(y_test.mean()), 4)}
    fitted, curves, probas = {}, {}, {}
    for name, pipe in build_models().items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, proba)
        m["cv_roc_auc_mean"] = round(float(cv_auc.mean()), 4)
        m["cv_roc_auc_std"] = round(float(cv_auc.std()), 4)
        results["models"][name] = m
        fitted[name] = pipe
        probas[name] = proba
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr)

    rf = fitted["Random Forest"]
    pred = (probas["Random Forest"] >= 0.5).astype(int)
    results["confusion_matrix_rf"] = confusion_matrix(y_test, pred).tolist()

    perm = permutation_importance(rf, X_test, y_test, scoring="roc_auc", n_repeats=15,
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # Out-of-fold probabilities for every lead (Random Forest).
    oof = cross_val_predict(build_models()["Random Forest"], df[FEATURES], df[config.TARGET],
                            cv=cv, method="predict_proba")[:, 1]
    lift = lift_table(df[config.TARGET], oof)
    results["oof_roc_auc_rf"] = round(float(roc_auc_score(df[config.TARGET], oof)), 4)
    results["top_decile_lift"] = round(float(lift.loc[0, "lift"]), 2)
    results["top3_deciles_capture"] = round(float(lift.loc[2, "cum_capture"]), 4)
    results["top_features"] = importance.head(8)[["feature", "importance"]].round(4).to_dict("records")

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, config.MODEL_PATH, compress=3)
        config.METRICS_PATH.write_text(json.dumps(results, indent=2))

    return {"results": results, "fitted": fitted, "curves": curves, "importance": importance,
            "lift": lift, "oof_proba": oof, "y_test": y_test, "test_proba": probas}


def load_model(path=config.MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_proba_row(model: Pipeline, row: dict) -> float:
    """Probability for one new lead given a dict with all FEATURES keys."""
    return float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
