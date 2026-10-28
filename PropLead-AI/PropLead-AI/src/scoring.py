"""Transparent, rule-based Lead Priority Score (0-100).

Why rule-based? Sales teams can read and challenge it. Every point of the score
can be traced back to one signal (see ``score_components``). No sensitive or
protected personal attribute (age, gender, religion, caste, nationality...) is used.

    score = 30*engagement + 20*site_visit + 15*recency + 15*responsiveness
            + 10*loan_ready + 10*followup          (each component scaled 0-1)

The score is an ESTIMATE to organise follow-ups, not a prediction of a sale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def score_components(df: pd.DataFrame) -> pd.DataFrame:
    """Points contributed by each component (sums to the priority score)."""
    w = config.SCORE_WEIGHTS
    unit = pd.DataFrame({
        "engagement": (df["total_engagement"] / config.ENGAGEMENT_CAP).clip(0, 1),
        "site_visit": df["site_visit_completed"].astype(float),
        "recency": 1 - (df["days_since_last_contact"] / config.RECENCY_CAP_DAYS).clip(0, 1),
        "responsiveness": 1 - (df["response_time_hours"] / config.RESPONSE_CAP_HOURS).clip(0, 1),
        "loan_ready": df["loan_preapproved"].astype(float),
        "followup": (df["followup_count"] / config.FOLLOWUP_CAP).clip(0, 1),
    }, index=df.index)
    return unit.mul(pd.Series(w)).round(2)


def assign_band(score: pd.Series) -> pd.Series:
    band = np.select(
        [score >= config.BAND_HIGH_MIN, score >= config.BAND_MEDIUM_MIN],
        ["High", "Medium"], default="Low",
    )
    return pd.Series(pd.Categorical(band, categories=config.BAND_ORDER, ordered=True), index=score.index)


def add_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comps = score_components(df)
    df["priority_score"] = comps.sum(axis=1).round(1)
    df["priority_band"] = assign_band(df["priority_score"])
    df["priority_rank"] = df["priority_score"].rank(method="first", ascending=False).astype(int)
    return df


def explain_lead(row: pd.Series) -> pd.DataFrame:
    """Per-component breakdown for one lead (used by the dashboard)."""
    comps = score_components(row.to_frame().T.infer_objects()).iloc[0]
    out = pd.DataFrame({"component": comps.index, "points": comps.values,
                        "max_points": [config.SCORE_WEIGHTS[c] for c in comps.index]})
    return out.sort_values("points", ascending=False).reset_index(drop=True)
