"""Feature engineering + lead-specific RFM-style (Recency / Frequency / Engagement) analysis.

The RFM idea is *adapted* to lead management; it is not classic purchase RFM:
  Recency    = days since last contact (fewer days -> better)
  Frequency  = number of follow-ups made on the lead
  Engagement = weighted sum of website visits, connected calls and WhatsApp replies
"""
from __future__ import annotations

import pandas as pd

from src import config


def _quintile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank-based 1-5 score (5 = best). Rank 'first' keeps bins equal despite many ties."""
    ranks = series.rank(method="first", ascending=higher_is_better)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = config.ENGAGEMENT_WEIGHTS

    df["total_engagement"] = sum(df[col] * weight for col, weight in w.items())
    df["interactions_total"] = df["calls_connected"] + df["whatsapp_replies"] + df["followup_count"]

    df["response_band"] = pd.cut(
        df["response_time_hours"], bins=config.RESPONSE_BINS,
        labels=config.RESPONSE_LABELS, include_lowest=True,
    )
    df["followup_intensity"] = pd.cut(
        df["followup_count"], bins=config.FOLLOWUP_BINS,
        labels=config.FOLLOWUP_LABELS, include_lowest=True,
    )
    df["budget_band"] = pd.cut(
        df["budget_lakh"], bins=config.BUDGET_BINS,
        labels=config.BUDGET_LABELS, include_lowest=True,
    )
    # 0 = neither, 1 = one of (loan pre-approved, site visit done), 2 = both
    df["readiness_level"] = df["loan_preapproved"] + df["site_visit_completed"]
    return df


def add_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, E scores (1-5), total (3-15) and an interpretable segment."""
    df = df.copy()
    df["R_score"] = _quintile_score(df["days_since_last_contact"], higher_is_better=False)
    df["F_score"] = _quintile_score(df["followup_count"])
    df["E_score"] = _quintile_score(df["total_engagement"])
    df["RFE_total"] = df["R_score"] + df["F_score"] + df["E_score"]
    df["rfe_segment"] = pd.cut(
        df["RFE_total"], bins=[0, 5, 8, 11, 15],
        labels=["Cold", "Cool", "Warm", "Hot"],
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_rfm(engineer_features(df))
