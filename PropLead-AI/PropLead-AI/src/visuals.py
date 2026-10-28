"""All static charts and project diagrams (saved to ./images).

Charts are produced from the pipeline outputs; diagrams are drawn with
matplotlib patches so that no external tool (Graphviz, draw.io) is needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config

P = config.PALETTE
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 11, "font.size": 10,
})


def _save(fig, name: str, out_dir: Path = config.IMAGES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _bar_labels(ax, fmt="{:.0f}", suffix=""):
    for c in ax.containers:
        ax.bar_label(c, labels=[fmt.format(v) + suffix for v in c.datavalues], padding=3, fontsize=9)


# --------------------------------------------------------------------------- #
# Data-quality & EDA charts
# --------------------------------------------------------------------------- #
def plot_missing_values(raw: pd.DataFrame) -> Path:
    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(miss.index, miss.values, color=P["accent"])
    ax.set_title("Missing values in raw data")
    ax.set_xlabel("Number of records")
    for i, v in enumerate(miss.values):
        ax.text(v + 0.3, i, f"{v} ({v / len(raw):.1%})", va="center", fontsize=9)
    ax.set_xlim(0, miss.max() * 1.3)
    return _save(fig, "01_missing_values.png")


def plot_leads_by_source(df: pd.DataFrame) -> Path:
    counts = df["lead_source"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=P["primary"])
    ax.set_title("Lead volume by source")
    ax.set_ylabel("Leads")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _bar_labels(ax)
    return _save(fig, "02_leads_by_source.png")


def plot_conversion_by_source(df: pd.DataFrame) -> Path:
    t = df.groupby("lead_source")["converted"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    overall = df["converted"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index, t["mean"] * 100, color=P["good"])
    ax.axhline(overall * 100, color=P["bad"], ls="--", lw=1.2, label=f"Overall {overall:.1%}")
    ax.set_title("Conversion rate by lead source")
    ax.set_ylabel("Conversion rate (%)")
    for i, (m, n) in enumerate(zip(t["mean"], t["size"])):
        ax.text(i, m * 100 + 0.6, f"{m:.1%}\n(n={n})", ha="center", fontsize=8.5)
    ax.set_ylim(0, t["mean"].max() * 100 * 1.25)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "03_conversion_by_source.png")


def plot_budget_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["budget_lakh"], bins=35, color=P["primary"], ax=axes[0])
    axes[0].set_title("Customer-stated budget distribution")
    axes[0].set_xlabel("Budget (INR lakh)")
    order = df.groupby("property_type")["budget_lakh"].median().sort_values().index
    sns.boxplot(data=df, x="property_type", y="budget_lakh", order=order, color=P["light"],
                linecolor=P["primary"], ax=axes[1])
    axes[1].set_title("Stated budget by property type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Budget (INR lakh)")
    plt.setp(axes[1].get_xticklabels(), rotation=15)
    return _save(fig, "04_budget_distribution.png")


def plot_city_property_heatmap(df: pd.DataFrame) -> Path:
    pt = df.pivot_table(index="city", columns="property_type", values="converted", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pt, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Conversion rate (%)"}, ax=ax)
    ax.set_title("Conversion rate (%) by city and property type\n(small cells are noisy)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _save(fig, "05_city_property_heatmap.png")


def plot_engagement_vs_conversion(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = pd.cut(df["total_engagement"], [-1, 5, 10, 15, 20, 100], labels=["0-5", "6-10", "11-15", "16-20", "21+"])
    r = df.groupby(bins, observed=True)["converted"].mean() * 100
    axes[0].bar(r.index.astype(str), r.values, color=P["primary"])
    axes[0].set_title("By total engagement score")
    axes[0].set_ylabel("Conversion rate (%)")
    s = df.groupby("site_visit_completed")["converted"].mean() * 100
    axes[1].bar(["No site visit", "Site visit done"], s.values, color=[P["grey"], P["good"]])
    axes[1].set_title("By site-visit status")
    l = df.groupby("loan_preapproved")["converted"].mean() * 100
    axes[2].bar(["Loan not pre-approved", "Loan pre-approved"], l.values, color=[P["grey"], P["good"]])
    axes[2].set_title("By loan pre-approval")
    for ax in axes:
        _bar_labels(ax, "{:.1f}", "%")
        ax.set_ylim(0, 60)
    return _save(fig, "06_engagement_vs_conversion.png")


def plot_response_time(df: pd.DataFrame) -> Path:
    t = df.groupby("response_band", observed=True)["converted"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t["mean"] * 100, color=P["accent"])
    ax.set_title("Conversion rate by first-response time")
    ax.set_xlabel("Response time band")
    ax.set_ylabel("Conversion rate (%)")
    for i, (m, n) in enumerate(zip(t["mean"], t["size"])):
        ax.text(i, m * 100 + 0.5, f"{m:.1%}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, t["mean"].max() * 100 * 1.3)
    return _save(fig, "07_response_time_vs_conversion.png")


def plot_correlation(df: pd.DataFrame) -> Path:
    cols = config.NUMERIC_FEATURES + [config.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix (numeric features + converted)")
    return _save(fig, "08_correlation_matrix.png")


def plot_rfe_segments(df: pd.DataFrame) -> Path:
    order = ["Cold", "Cool", "Warm", "Hot"]
    t = df.groupby("rfe_segment", observed=True).agg(leads=("converted", "size"), rate=("converted", "mean")).reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(order, t["leads"], color=P["primary"])
    axes[0].set_title("Leads per RFE segment")
    _bar_labels(axes[0])
    axes[1].bar(order, t["rate"] * 100, color=[P["grey"], "#E0A339", P["accent"], P["bad"]])
    axes[1].set_title("Conversion rate per RFE segment")
    axes[1].set_ylabel("%")
    _bar_labels(axes[1], "{:.1f}", "%")
    return _save(fig, "09_rfe_segments.png")


# --------------------------------------------------------------------------- #
# Priority score charts
# --------------------------------------------------------------------------- #
def plot_score_weights() -> Path:
    w = pd.Series(config.SCORE_WEIGHTS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(w.index, w.values, color=P["primary"])
    ax.set_title("Lead priority score: points per component (max 100)")
    ax.set_xlabel("Maximum points")
    for i, v in enumerate(w.values):
        ax.text(v + 0.5, i, str(v), va="center")
    ax.set_xlim(0, 36)
    return _save(fig, "10_priority_score_weights.png")


def plot_priority_bands(df: pd.DataFrame) -> Path:
    t = df.groupby("priority_band", observed=True).agg(leads=("converted", "size"), rate=("converted", "mean"))
    order = config.BAND_ORDER
    t = t.reindex(order)
    colors = [config.BAND_COLORS[b] for b in order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(df["priority_score"], bins=30, color=P["primary"], ax=axes[0])
    axes[0].axvline(config.BAND_MEDIUM_MIN, color=P["grey"], ls="--")
    axes[0].axvline(config.BAND_HIGH_MIN, color=P["grey"], ls="--")
    axes[0].set_title("Priority score distribution")
    axes[0].set_xlabel("Score (0-100)")
    axes[1].bar(order, t["leads"], color=colors)
    axes[1].set_title("Leads per priority band")
    _bar_labels(axes[1])
    axes[2].bar(order, t["rate"] * 100, color=colors)
    axes[2].set_title("Observed conversion rate per band")
    axes[2].set_ylabel("%")
    _bar_labels(axes[2], "{:.1f}", "%")
    return _save(fig, "11_priority_bands.png")


# --------------------------------------------------------------------------- #
# Channel analysis
# --------------------------------------------------------------------------- #
def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("lead_source")
    t = pd.DataFrame({
        "leads": g.size(),
        "conversions": g["converted"].sum(),
        "conversion_rate": g["converted"].mean(),
        "median_budget_lakh": g["budget_lakh"].median(),
        "total_stated_budget_cr": g["budget_lakh"].sum() / 100,
        "avg_priority_score": g["priority_score"].mean(),
        "high_priority_share": g["priority_band"].apply(lambda s: (s == "High").mean()),
        "avg_response_hours": g["response_time_hours"].mean(),
    }).sort_values("conversion_rate", ascending=False)
    return t.round(3)


def plot_channel_overview(df: pd.DataFrame) -> Path:
    t = channel_summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(t.index, t["leads"], color=P["primary"])
    axes[0].set_title("Lead volume")
    axes[1].bar(t.index, t["conversion_rate"] * 100, color=P["good"])
    axes[1].set_title("Conversion rate (%)")
    axes[2].bar(t.index, t["total_stated_budget_cr"], color=P["accent"])
    axes[2].set_title("Total customer-stated budget (INR crore)\n(not company revenue)")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _bar_labels(axes[0])
    _bar_labels(axes[1], "{:.1f}", "%")
    _bar_labels(axes[2], "{:.0f}")
    return _save(fig, "12_channel_overview.png")


# --------------------------------------------------------------------------- #
# Model charts
# --------------------------------------------------------------------------- #
def plot_roc(curves: dict, metrics: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = [P["grey"], P["accent"], P["primary"]]
    for (name, (fpr, tpr)), c in zip(curves.items(), colors):
        auc = metrics["models"][name]["roc_auc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="black", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (hold-out test set)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return _save(fig, "13_roc_curves.png")


def plot_confusion(cm: list) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"], ax=ax)
    ax.set_title("Random Forest confusion matrix\n(hold-out, threshold 0.5)")
    return _save(fig, "14_confusion_matrix.png")


def plot_feature_importance(importance: pd.DataFrame) -> Path:
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=P["primary"], ecolor=P["grey"])
    ax.set_title("Permutation importance (drop in ROC-AUC)\nRandom Forest, hold-out set")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    return _save(fig, "15_feature_importance.png")


def plot_lift(lift: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(lift["decile"].astype(str), lift["conv_rate"] * 100, color=P["primary"])
    base = lift["conversions"].sum() / lift["leads"].sum() * 100
    axes[0].axhline(base, color=P["bad"], ls="--", label=f"Average {base:.1f}%")
    axes[0].set_title("Conversion rate by predicted-probability decile\n(out-of-fold, decile 1 = highest)")
    axes[0].set_xlabel("Decile")
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False)
    axes[1].plot([0] + list(range(1, 11)), [0] + list(lift["cum_capture"] * 100), marker="o", color=P["good"], label="Model")
    axes[1].plot([0, 10], [0, 100], ls=":", color="black", label="Random")
    axes[1].set_title("Cumulative share of conversions captured")
    axes[1].set_xlabel("Deciles contacted (best first)")
    axes[1].set_ylabel("% of all conversions")
    axes[1].legend(frameon=False)
    return _save(fig, "16_lift_gains.png")


def plot_model_comparison(metrics: dict) -> Path:
    t = pd.DataFrame(metrics["models"]).T[["precision", "recall", "f1", "roc_auc"]]
    ax = t.plot(kind="bar", figsize=(9, 4.2), color=[P["primary"], P["accent"], P["good"], P["grey"]], width=0.8)
    ax.set_title("Model comparison on hold-out test set")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7.5, padding=2)
    return _save(ax.get_figure(), "17_model_comparison.png")


# --------------------------------------------------------------------------- #
# Diagrams
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc, ec="#1F4E79", fs=9, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))


def diagram_workflow() -> Path:
    steps = [
        ("1. Data\nCollection", "Synthetic CSV\n(1,200 leads)"),
        ("2. Data\nCleaning", "Duplicates, missing,\ncategories, outliers"),
        ("3. EDA", "Source mix, budget,\ncity, property type"),
        ("4. Feature\nEngineering", "Engagement, response\nbands, follow-up"),
        ("5. RFE\nAnalysis", "Recency, Frequency,\nEngagement"),
        ("6. Priority\nScore", "Explainable 0-100\nLow / Medium / High"),
        ("7. Channel\nAnalysis", "Volume, conversion,\nstated budget"),
        ("8. Conversion\nModel", "Baseline, LogReg,\nRandom Forest"),
        ("9. Dashboard", "Streamlit app\n(5 tabs)"),
        ("10. Business\nRecommendations", "Follow-up order,\nchannel tests"),
    ]
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("PropLead AI - end-to-end workflow", fontsize=15, fontweight="bold", pad=10)
    w, h = 2.4, 1.0
    xs = [0.5 + i * 2.85 for i in range(5)]
    for row, (y, order) in enumerate([(4.3, range(0, 5)), (1.5, range(9, 4, -1))]):
        for pos, idx in enumerate(order):
            x = xs[pos]
            title, sub = steps[idx]
            _box(ax, x, y, w, h, title, P["primary"], fs=10, tc="white", bold=True)
            _box(ax, x, y - 0.95, w, 0.75, sub, P["light"], fs=8.5)
            if row == 0 and pos < 4:
                _arrow(ax, x + w, y + h / 2, xs[pos + 1], y + h / 2)
            if row == 1 and pos > 0:  # bottom row flows right -> left
                _arrow(ax, x, y + h / 2, xs[pos - 1] + w, y + h / 2)
    _arrow(ax, xs[4] + w / 2, 4.3 - 0.95, xs[4] + w / 2, 1.5 + h + 0.02)
    ax.text(7.5, 0.12, "Human review stays in the loop: scores are decision support, not guarantees.",
            ha="center", fontsize=10, style="italic", color=P["grey"])
    return _save(fig, "diagram_workflow.png")


def diagram_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("PropLead AI - system architecture", fontsize=15, fontweight="bold", pad=10)

    layers = [
        (6.2, "DATA LAYER", "#D6E4F0", [
            ("src/generate_dataset.py\nSynthetic lead generator", 0.6),
            ("data/sample_real_estate_leads.csv\nRaw leads (1,200)", 4.9),
            ("Future: anonymised CRM export\n(consented, no PII)", 9.2)]),
        (4.4, "PROCESSING LAYER", "#DDEFE3", [
            ("src/data_cleaning.py\nValidate, dedupe, impute", 0.6),
            ("src/features.py\nEngagement, bands, RFE", 4.9),
            ("src/scoring.py\nExplainable priority score", 9.2)]),
        (2.6, "ANALYTICS / ML LAYER", "#FBE8D5", [
            ("src/model.py\nLogReg + Random Forest\nCV, hold-out, lift", 0.6),
            ("src/visuals.py\nCharts & diagrams", 4.9),
            ("src/generate_report.py\nData-driven recommendations", 9.2)]),
        (0.8, "PRESENTATION LAYER", "#F1DDF0", [
            ("app.py (Streamlit + Plotly)\nOverview | Channels | Priority\nModel | Score a lead", 0.6),
            ("notebooks/\nEDA walkthrough", 4.9),
            ("reports/ + images/ + README\nGitHub deliverables", 9.2)]),
    ]
    for y, label, fc, boxes in layers:
        ax.add_patch(FancyBboxPatch((0.2, y - 0.15), 13.6, 1.45, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=fc, ec="#999999", lw=1, alpha=0.55))
        ax.text(0.35, y + 1.12, label, fontsize=9, fontweight="bold", color="#333333")
        for text, x in boxes:
            _box(ax, x + 0.1, y + 0.05, 3.9, 0.95, text, "white", fs=8.5)
    for x in (2.7, 7.0, 11.3):
        for y in (6.05, 4.25, 2.45):
            _arrow(ax, x, y, x, y - 0.25 + 0.05)
    return _save(fig, "diagram_architecture.png")


def diagram_data_model() -> Path:
    groups = {
        "Identity": ["lead_id"],
        "Acquisition": ["lead_source", "campaign_name", "city"],
        "Property requirement": ["property_type", "preferred_bhk", "budget_lakh"],
        "Engagement": ["website_visits", "calls_connected", "whatsapp_replies"],
        "Sales activity": ["site_visit_completed", "followup_count", "response_time_hours",
                           "days_since_last_contact"],
        "Readiness": ["loan_preapproved", "discount_interest"],
        "Outcome (target)": ["lead_status*", "converted"],
    }
    fig, ax = plt.subplots(figsize=(13, 6.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    ax.set_title("Dataset field groups (18 columns)", fontsize=14, fontweight="bold")
    colors = ["#D6E4F0", "#DDEFE3", "#FBE8D5", "#F1DDF0", "#FFF3C4", "#E3F2F1", "#F8D7D7"]
    pos = [(0.3, 6.0), (3.5, 6.0), (6.7, 6.0), (9.9, 6.0), (0.3, 3.1), (4.1, 3.1), (7.9, 3.1)]  # (x, top)
    for (name, cols), (x, top), c in zip(groups.items(), pos, colors):
        h = 0.5 + 0.38 * len(cols)
        y = top - (h + 0.55)
        ax.add_patch(FancyBboxPatch((x, y), 2.9, h + 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c, ec="#555555", lw=1.2))
        ax.text(x + 1.45, top - 0.3, name, ha="center", fontsize=10, fontweight="bold")
        for i, col in enumerate(cols):
            ax.text(x + 0.2, top - 0.75 - i * 0.38, f"- {col}", fontsize=9, family="monospace")
    ax.text(0.3, 0.05, "* lead_status is recorded after the outcome -> excluded from the model to avoid target leakage.",
            fontsize=9, style="italic", color=P["bad"])
    return _save(fig, "diagram_data_model.png")


def diagram_scoring_logic() -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.7, 5)
    ax.axis("off")
    ax.set_title("How the lead priority score is built", fontsize=14, fontweight="bold")
    comps = [("Engagement\n(visits, calls x2, WhatsApp)", "30"), ("Site visit done", "20"),
             ("Recency\n(days since contact)", "15"), ("Responsiveness\n(first response hrs)", "15"),
             ("Loan pre-approved", "10"), ("Follow-up count", "10")]
    for i, (t, pts) in enumerate(comps):
        y = 4.0 - i * 0.72
        _box(ax, 0.3, y - 0.28, 3.9, 0.6, t, P["light"], fs=9)
        _box(ax, 4.5, y - 0.28, 0.9, 0.6, f"{pts} pts", "white", fs=9, bold=True)
        _arrow(ax, 5.4, y, 6.6, 2.4)
    _box(ax, 6.7, 1.9, 2.2, 1.0, "Priority score\n0 - 100", P["primary"], fs=11, tc="white", bold=True)
    _arrow(ax, 8.9, 2.4, 9.6, 2.4)
    for (t, y, c) in [("HIGH  (>= 55)", 3.7, config.BAND_COLORS["High"]),
                      ("MEDIUM  (40 - 54.9)", 2.4, config.BAND_COLORS["Medium"]),
                      ("LOW  (< 40)", 1.1, config.BAND_COLORS["Low"])]:
        _box(ax, 9.7, y - 0.35, 3.0, 0.7, t, c, ec=c, fs=10, tc="white", bold=True)
        _arrow(ax, 9.6, 2.4, 9.7, y, color=c)
    ax.text(6.5, -0.45, "No sensitive personal attributes are used. Weights are editable in src/config.py.",
            ha="center", fontsize=9.5, style="italic", color=P["grey"])
    return _save(fig, "diagram_scoring_logic.png")


def diagram_dashboard_wireframe() -> Path:
    """Wireframe (design sketch) of the Streamlit app - NOT a screenshot."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Dashboard wireframe (design sketch, not a screenshot)", fontsize=14, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 12.6, 6.1, boxstyle="round,pad=0.02", fc="white", ec="#444", lw=1.5))
    _box(ax, 0.4, 0.4, 2.6, 5.7, "SIDEBAR\n\nFilters:\n- Lead source\n- City\n- Property type\n- Budget range\n\nCost per lead\n(assumption inputs)", "#EEF2F7", fs=9)
    for i, t in enumerate(["Overview", "Channels", "Lead Priority", "Model", "Score a lead"]):
        _box(ax, 3.2 + i * 1.9, 5.5, 1.8, 0.55, t, P["primary"] if i == 0 else "#DDDDDD", fs=9,
             tc="white" if i == 0 else "black", bold=True)
    for i, t in enumerate(["Total leads", "Converted", "Conversion rate", "High-priority leads"]):
        _box(ax, 3.2 + i * 2.4, 4.3, 2.2, 0.9, f"KPI\n{t}", P["light"], fs=9)
    _box(ax, 3.2, 0.6, 4.6, 3.4, "Chart: conversion rate\nby lead source", "white", fs=10)
    _box(ax, 8.0, 2.3, 4.6, 1.7, "Chart: leads per priority band", "white", fs=10)
    _box(ax, 8.0, 0.6, 4.6, 1.5, "Table: top 20 leads to call first\n(score, band, probability)", "white", fs=10)
    return _save(fig, "diagram_dashboard_wireframe.png")


def make_all_diagrams() -> list[Path]:
    return [diagram_workflow(), diagram_architecture(), diagram_data_model(),
            diagram_scoring_logic(), diagram_dashboard_wireframe(), plot_score_weights()]
