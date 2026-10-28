"""Central configuration: paths, constants and scoring rules for PropLead AI.

Everything that a reviewer may want to tune (score weights, band thresholds,
random seed) lives here so that the scoring logic stays transparent.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "sample_real_estate_leads.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_PATH = PROCESSED_DIR / "cleaned_leads.csv"
SCORED_PATH = PROCESSED_DIR / "scored_leads.csv"
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_conversion_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.json"

# --------------------------------------------------------------------------- #
# Reproducibility / dataset
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_LEADS = 1200
TARGET_CONVERSION_RATE = 0.22  # overall rate the *synthetic* generator aims for

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
# total_engagement = visits*W_VISIT + connected calls*W_CALL + WhatsApp replies*W_WA
ENGAGEMENT_WEIGHTS = {"website_visits": 1, "calls_connected": 2, "whatsapp_replies": 1}

RESPONSE_BINS = [0, 2, 6, 24, float("inf")]
RESPONSE_LABELS = ["0-2h", "2-6h", "6-24h", "24h+"]

FOLLOWUP_BINS = [0, 2, 5, float("inf")]
FOLLOWUP_LABELS = ["Low (1-2)", "Medium (3-5)", "High (6+)"]

BUDGET_BINS = [0, 50, 100, 200, float("inf")]
BUDGET_LABELS = ["<50L", "50-100L", "100-200L", "200L+"]

# --------------------------------------------------------------------------- #
# Explainable lead-priority score (0-100)
# Each component is scaled to 0-1 with a documented cap, then multiplied by its
# weight. Weights sum to 100 so the score reads like "points".
# --------------------------------------------------------------------------- #
SCORE_WEIGHTS = {
    "engagement": 30,      # total_engagement / ENGAGEMENT_CAP
    "site_visit": 20,      # 1 if a site visit has been completed
    "recency": 15,         # fresher contact -> higher
    "responsiveness": 15,  # faster first response -> higher
    "loan_ready": 10,      # 1 if loan is pre-approved
    "followup": 10,        # followup_count / FOLLOWUP_CAP
}
ENGAGEMENT_CAP = 25.0
RECENCY_CAP_DAYS = 60.0
RESPONSE_CAP_HOURS = 48.0
FOLLOWUP_CAP = 6.0

# Priority bands (score >= threshold). Calibrated once on the demo data and then
# frozen, so that bands are a documented rule rather than a moving target.
BAND_HIGH_MIN = 55.0
BAND_MEDIUM_MIN = 40.0
BAND_ORDER = ["Low", "Medium", "High"]

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "preferred_bhk",
    "budget_lakh",
    "website_visits",
    "calls_connected",
    "whatsapp_replies",
    "followup_count",
    "response_time_hours",
    "days_since_last_contact",
    "total_engagement",
    "site_visit_completed",
    "loan_preapproved",
    "discount_interest",
]
CATEGORICAL_FEATURES = ["lead_source", "city", "property_type"]

# Never used as model inputs:
#   lead_id        -> identifier, no signal
#   lead_status    -> recorded AFTER the outcome is known (target leakage)
#   converted      -> the target itself
#   campaign_name  -> nested inside lead_source, excluded to keep the model simple
EXCLUDED_FROM_MODEL = ["lead_id", "lead_status", "converted", "campaign_name"]
TARGET = "converted"

# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
PALETTE = {
    "primary": "#1F4E79",
    "accent": "#E07B39",
    "good": "#2E8B57",
    "bad": "#C0392B",
    "grey": "#7F8C8D",
    "light": "#EAF1F8",
}
BAND_COLORS = {"Low": "#C0392B", "Medium": "#E0A339", "High": "#2E8B57"}
