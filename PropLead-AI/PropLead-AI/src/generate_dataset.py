"""Generate the SYNTHETIC real-estate lead dataset used by PropLead AI.

IMPORTANT
---------
All records are artificially generated for demonstration. They are NOT real
customers and the relationships inside the data were written into the generator
by the author. Results therefore demonstrate the *workflow and code*, and must
never be presented as evidence of real real-estate market behaviour.

Usage
-----
    python -m src.generate_dataset                # 1,200 rows -> data/sample_real_estate_leads.csv
    python -m src.generate_dataset --rows 5000 --seed 7 --out data/big.csv
    python -m src.generate_dataset --clean        # no deliberately injected quality issues
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# Generator assumptions (all documented, all editable)
# --------------------------------------------------------------------------- #
SOURCE_SHARE = {
    "Property Portal": 0.28,
    "Google Ads": 0.22,
    "Instagram": 0.16,
    "Website": 0.14,
    "Referral": 0.11,
    "Walk-in": 0.09,
}
# Additive effect of the source on the conversion log-odds.
SOURCE_EFFECT = {
    "Property Portal": 0.00,
    "Google Ads": -0.10,
    "Instagram": -0.50,
    "Website": 0.10,
    "Referral": 0.80,
    "Walk-in": 0.90,
}
CAMPAIGNS = {
    "Property Portal": ["Portal_Premium_Listing", "Portal_Featured_Project", "Portal_Basic_Listing"],
    "Google Ads": ["GAds_Search_Brand", "GAds_Search_Generic", "GAds_Display_Remarketing"],
    "Instagram": ["IG_Reels_Showcase", "IG_Lead_Form", "IG_Influencer_Tour"],
    "Website": ["Web_Organic", "Web_Brochure_Download", "Web_Chatbot"],
    "Referral": ["Ref_Existing_Customer", "Ref_Broker_Network"],
    "Walk-in": ["Walkin_Sales_Office", "Walkin_Site_Hoarding"],
}
CITY_SHARE = {
    "Noida": 0.17, "Gurugram": 0.14, "Ghaziabad": 0.10, "Lucknow": 0.10,
    "Jaipur": 0.09, "Pune": 0.14, "Bengaluru": 0.14, "Hyderabad": 0.12,
}
CITY_PRICE_FACTOR = {
    "Noida": 1.10, "Gurugram": 1.40, "Ghaziabad": 0.85, "Lucknow": 0.80,
    "Jaipur": 0.80, "Pune": 1.15, "Bengaluru": 1.30, "Hyderabad": 1.20,
}
PROPERTY_SHARE = {
    "Apartment": 0.46, "Builder Floor": 0.16, "Villa": 0.14, "Plot": 0.14, "Commercial": 0.10,
}
PROPERTY_MEDIAN_LAKH = {
    "Apartment": 75, "Builder Floor": 95, "Villa": 180, "Plot": 55, "Commercial": 120,
}
BHK_OPTIONS = {
    "Apartment": ([1, 2, 3, 4], [0.10, 0.40, 0.38, 0.12]),
    "Builder Floor": ([2, 3, 4], [0.30, 0.50, 0.20]),
    "Villa": ([3, 4, 5], [0.40, 0.45, 0.15]),
}  # Plot / Commercial -> 0 (not applicable)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisection: find b so that mean(sigmoid(z + b)) == target_rate."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(
    n: int = config.N_LEADS,
    seed: int = config.RANDOM_STATE,
    target_rate: float = config.TARGET_CONVERSION_RATE,
    inject_issues: bool = True,
) -> pd.DataFrame:
    """Create the synthetic lead table."""
    rng = np.random.default_rng(seed)

    lead_source = rng.choice(list(SOURCE_SHARE), size=n, p=list(SOURCE_SHARE.values()))
    campaign = np.array([rng.choice(CAMPAIGNS[s]) for s in lead_source])
    city = rng.choice(list(CITY_SHARE), size=n, p=list(CITY_SHARE.values()))
    prop = rng.choice(list(PROPERTY_SHARE), size=n, p=list(PROPERTY_SHARE.values()))

    bhk = np.array([
        int(rng.choice(BHK_OPTIONS[p][0], p=BHK_OPTIONS[p][1])) if p in BHK_OPTIONS else 0
        for p in prop
    ])
    budget = np.array([
        PROPERTY_MEDIAN_LAKH[p] * CITY_PRICE_FACTOR[c] * rng.lognormal(0, 0.35)
        for p, c in zip(prop, city)
    ]).round(1)

    # Latent buying "intent" - drives engagement AND conversion (hidden from the data).
    intent = rng.normal(0, 1, n)
    is_walkin = lead_source == "Walk-in"
    is_ref = lead_source == "Referral"
    is_insta = lead_source == "Instagram"

    website_visits = rng.poisson(np.exp(1.1 + 0.30 * intent - 0.9 * is_walkin)).clip(0, 30)
    calls = rng.poisson(np.exp(0.2 + 0.40 * intent + 0.4 * is_ref + 0.3 * is_walkin)).clip(0, 8)
    whatsapp = rng.poisson(np.exp(0.3 + 0.35 * intent + 0.3 * is_insta)).clip(0, 10)
    followups = (1 + rng.poisson(np.exp(0.7 + 0.15 * intent))).clip(1, 12)
    response_h = np.exp(np.log(6) - 0.25 * intent + rng.normal(0, 0.8, n)).clip(0.1, 72).round(1)
    days_since = rng.exponential(np.exp(2.0 - 0.2 * intent)).round().clip(0, 90).astype(int)

    src_eff = pd.Series(lead_source).map(SOURCE_EFFECT).to_numpy()
    loan = rng.random(n) < _sigmoid(-0.6 + 0.6 * intent)
    discount = rng.random(n) < 0.45
    site_visit = rng.random(n) < _sigmoid(-1.3 + 0.8 * intent + 0.15 * calls + 0.2 * src_eff)

    # Conversion log-odds (the "ground truth" the model will try to recover).
    z = (
        src_eff
        + 0.30 * intent
        + 1.10 * site_visit
        + 0.18 * calls
        + 0.10 * whatsapp
        + 0.04 * website_visits
        + 0.60 * loan
        - 0.10 * discount
        - 0.025 * np.minimum(response_h, 48)
        - 0.020 * days_since
        + 0.05 * followups
        + rng.normal(0, 0.5, n)
    )
    z = z + _solve_intercept(z, target_rate)
    converted = (rng.random(n) < _sigmoid(z)).astype(int)

    # lead_status is recorded AFTER the outcome -> it is a leakage column by design.
    status = np.empty(n, dtype=object)
    for i in range(n):
        if converted[i]:
            status[i] = "Booked"
        elif site_visit[i]:
            status[i] = rng.choice(["Lost", "Nurturing", "Negotiation"], p=[0.40, 0.30, 0.30])
        else:
            status[i] = rng.choice(["Lost", "Nurturing", "New / Uncontacted"], p=[0.45, 0.40, 0.15])

    df = pd.DataFrame({
        "lead_id": [f"L{i:04d}" for i in range(1, n + 1)],
        "lead_source": lead_source,
        "campaign_name": campaign,
        "city": city,
        "property_type": prop,
        "preferred_bhk": bhk,
        "budget_lakh": budget,
        "website_visits": website_visits,
        "calls_connected": calls,
        "whatsapp_replies": whatsapp,
        "site_visit_completed": site_visit.astype(int),
        "followup_count": followups,
        "response_time_hours": response_h,
        "days_since_last_contact": days_since,
        "loan_preapproved": loan.astype(int),
        "discount_interest": discount.astype(int),
        "lead_status": status,
        "converted": converted,
    })

    if inject_issues:
        df = _inject_quality_issues(df, rng)
    return df


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a few realistic data-quality problems so the cleaning step has work to do."""
    df = df.copy()
    n = len(df)

    # Missing values
    df.loc[rng.choice(n, int(0.025 * n), replace=False), "response_time_hours"] = np.nan
    df.loc[rng.choice(n, int(0.015 * n), replace=False), "budget_lakh"] = np.nan
    df.loc[rng.choice(n, int(0.010 * n), replace=False), "city"] = np.nan

    # Inconsistent category spellings
    variants = {
        "Google Ads": ["google ads", "Google ads ", " GOOGLE ADS"],
        "Walk-in": ["walk in", "Walkin", "WALK-IN"],
        "Instagram": ["instagram", "Insta"],
        "Property Portal": ["property portal", "Portal"],
        "Referral": ["referral", "Referal"],
        "Website": ["website", "web site"],
    }
    idx = rng.choice(n, int(0.04 * n), replace=False)
    for i in idx:
        canon = df.at[i, "lead_source"]
        df.at[i, "lead_source"] = rng.choice(variants[canon])

    for i in rng.choice(n, 10, replace=False):
        if df.at[i, "city"] == "Gurugram":
            df.at[i, "city"] = "Gurgaon"
        elif df.at[i, "city"] == "Bengaluru":
            df.at[i, "city"] = "Bangalore"
        elif isinstance(df.at[i, "city"], str):
            df.at[i, "city"] = df.at[i, "city"].upper()

    for i in rng.choice(n, 8, replace=False):
        df.at[i, "property_type"] = str(df.at[i, "property_type"]).lower() + " "
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=config.N_LEADS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=config.RAW_PATH)
    parser.add_argument("--clean", action="store_true", help="do not inject data-quality issues")
    args = parser.parse_args()

    df = generate(args.rows, args.seed, inject_issues=not args.clean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic leads -> {args.out}")
    print(f"Overall conversion rate: {df['converted'].mean():.1%}")


if __name__ == "__main__":
    main()
