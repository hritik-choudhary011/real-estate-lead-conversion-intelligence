"""Data cleaning for the PropLead AI lead table.

Steps (all counted in the returned report):
  1. strip whitespace and standardise category spellings
  2. remove exact / lead_id duplicates
  3. validate ranges and binary fields
  4. impute missing values (median by group) and keep *_imputed flags
  5. cap extreme outliers at the 99th percentile (winsorise)

Known limitation: medians are computed on the whole table before the
train/test split. With ~2% missing values the effect is negligible, but a
production system should fit imputers inside the modelling pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

SOURCE_MAP = {
    "propertyportal": "Property Portal", "portal": "Property Portal",
    "googleads": "Google Ads",
    "instagram": "Instagram", "insta": "Instagram",
    "website": "Website",
    "referral": "Referral", "referal": "Referral",
    "walkin": "Walk-in",
}
CITY_MAP = {
    "noida": "Noida", "gurugram": "Gurugram", "gurgaon": "Gurugram",
    "ghaziabad": "Ghaziabad", "lucknow": "Lucknow", "jaipur": "Jaipur",
    "pune": "Pune", "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "hyderabad": "Hyderabad",
}
PROPERTY_MAP = {
    "apartment": "Apartment", "builderfloor": "Builder Floor", "villa": "Villa",
    "plot": "Plot", "commercial": "Commercial",
}
BINARY_COLS = ["site_visit_completed", "loan_preapproved", "discount_interest", "converted"]
COUNT_COLS = ["website_visits", "calls_connected", "whatsapp_replies", "followup_count",
              "days_since_last_contact", "preferred_bhk"]


def _key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _standardise(series: pd.Series, mapping: dict, name: str) -> tuple[pd.Series, int]:
    """Map messy spellings to canonical labels; unknown labels are kept stripped."""
    original = series.copy()
    mapped = series.map(lambda v: mapping.get(_key(v), str(v).strip()) if pd.notna(v) else v)
    changed = int(((mapped != original) & original.notna()).sum())
    return mapped, changed


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned_df, report). The input frame is not modified."""
    df = df.copy()
    report: dict = {"rows_in": int(len(df))}
    report["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    # 1. text normalisation ------------------------------------------------- #
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
    fixes = {}
    df["lead_source"], fixes["lead_source"] = _standardise(df["lead_source"], SOURCE_MAP, "lead_source")
    df["city"], fixes["city"] = _standardise(df["city"], CITY_MAP, "city")
    df["property_type"], fixes["property_type"] = _standardise(df["property_type"], PROPERTY_MAP, "property_type")
    report["category_fixes"] = fixes

    # 2. duplicates --------------------------------------------------------- #
    n0 = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = int(n0 - len(df))
    n1 = len(df)
    df = df.drop_duplicates(subset="lead_id", keep="first")
    report["duplicate_lead_ids_removed"] = int(n1 - len(df))

    # 3. validation --------------------------------------------------------- #
    issues = {}
    for col in BINARY_COLS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            issues[f"{col}_not_binary"] = int(bad.sum())
            df = df[~bad]
    for col in COUNT_COLS + ["budget_lakh", "response_time_hours"]:
        neg = (df[col] < 0).sum()
        if neg:
            issues[f"{col}_negative"] = int(neg)
            df.loc[df[col] < 0, col] = np.nan
    report["validation_issues"] = issues

    # 4. missing values ----------------------------------------------------- #
    df["city"] = df["city"].fillna("Unknown")
    df["response_time_imputed"] = df["response_time_hours"].isna().astype(int)
    df["budget_imputed"] = df["budget_lakh"].isna().astype(int)
    df["response_time_hours"] = df["response_time_hours"].fillna(
        df.groupby("lead_source")["response_time_hours"].transform("median")
    ).fillna(df["response_time_hours"].median())
    df["budget_lakh"] = df["budget_lakh"].fillna(
        df.groupby(["property_type", "city"])["budget_lakh"].transform("median")
    ).fillna(df.groupby("property_type")["budget_lakh"].transform("median"))
    report["missing_after"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    # 5. outliers (IQR flag, 99th-percentile cap) --------------------------- #
    capped = {}
    for col in ["budget_lakh", "response_time_hours", "website_visits", "days_since_last_contact"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        fence = q3 + 1.5 * (q3 - q1)
        cap = df[col].quantile(0.99)
        capped[col] = {
            "iqr_upper_fence": round(float(fence), 2),
            "n_above_fence": int((df[col] > fence).sum()),
            "cap_p99": round(float(cap), 2),
            "n_capped": int((df[col] > cap).sum()),
        }
        df[col] = df[col].clip(upper=cap)
    report["outliers"] = capped

    for col in COUNT_COLS + BINARY_COLS:
        df[col] = df[col].astype(int)
    df["budget_lakh"] = df["budget_lakh"].round(1)
    df["response_time_hours"] = df["response_time_hours"].round(1)

    report["rows_out"] = int(len(df))
    report["unique_lead_sources"] = sorted(df["lead_source"].unique().tolist())
    return df.reset_index(drop=True), report


def load_raw(path: Path = config.RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_and_save(raw_path: Path = config.RAW_PATH) -> tuple[pd.DataFrame, dict]:
    df, report = clean(load_raw(raw_path))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_PATH, index=False)
    config.CLEANING_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return df, report


if __name__ == "__main__":
    _, rep = clean_and_save()
    print(json.dumps(rep, indent=2))
