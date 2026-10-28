"""Build and execute notebooks/PropLead_AI_Analysis.ipynb WITHOUT needing Jupyter.

The notebook is written as plain nbformat-4 JSON; cells are executed with a tiny
in-process runner that stores stdout, DataFrame tables and matplotlib figures as
cell outputs so the notebook renders fully on GitHub.

If you have Jupyter installed you can simply open the notebook and "Run all"
instead - this script is only a convenience for regenerating it.

    python scripts/build_notebook.py
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "PropLead_AI_Analysis.ipynb"

MD, CODE = "markdown", "code"
cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append((MD, text.strip("\n")))


def code(text: str) -> None:
    cells.append((CODE, text.strip("\n")))


# --------------------------------------------------------------------------- #
md("""
# PropLead AI - Real Estate Lead Conversion Intelligence System
**Analysis notebook** | PGDM - Marketing + Business Analytics

> ⚠️ **Data caveat:** the dataset is **synthetic** (1,200 generated records). Results demonstrate the *workflow and code*;
> they are **not** evidence of real real-estate market behaviour. Replace with authorised, anonymised CRM data for real use.

**Notebook map:** 1 Setup → 2 Load & profile → 3 Cleaning → 4 EDA → 5 Feature engineering → 6 RFE analysis →
7 Priority score → 8 Channel analysis → 9 Conversion model → 10 Recommendations → 11 Limitations
""")

md("## 1. Setup")
code("""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config, data_cleaning, features, scoring, model, visuals
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40)
print("Project root:", ROOT)
""")

md("## 2. Load and profile the raw data")
code("""
raw = data_cleaning.load_raw()
print("Shape:", raw.shape)
raw.head()
""")
code("""
print(raw.dtypes)
print()
print("Missing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0])
print()
print("Distinct spellings in lead_source (before cleaning):", raw["lead_source"].nunique())
print(raw["lead_source"].value_counts().tail(8))
""")

md("""
## 3. Data cleaning
`src/data_cleaning.py` standardises categories, removes duplicates, validates ranges, imputes missing values
(median by group, with `*_imputed` flags) and caps extreme outliers at the 99th percentile.
""")
code("""
df, report = data_cleaning.clean(raw)
import json
print(json.dumps({k: report[k] for k in ["rows_in", "rows_out", "missing_before", "category_fixes",
                                          "exact_duplicates_removed", "duplicate_lead_ids_removed",
                                          "validation_issues", "missing_after"]}, indent=2))
print("Clean lead sources:", report["unique_lead_sources"])
""")
code("""
fig, ax = plt.subplots(figsize=(6.5, 3))
miss = raw.isna().sum(); miss = miss[miss > 0].sort_values()
ax.barh(miss.index, miss.values, color="#E07B39"); ax.set_title("Missing values in raw data"); ax.set_xlabel("Records")
plt.show()
""")

md("## 4. Exploratory data analysis (EDA)")
code("""
df = features.build_features(df)
print(f"Leads: {len(df):,} | Conversions: {df['converted'].sum()} | Overall conversion rate: {df['converted'].mean():.1%}")
df.describe().T.round(2)
""")
md("### 4.1 Lead volume and conversion by source")
code("""
src_tbl = df.groupby("lead_source")["converted"].agg(leads="size", conversions="sum", conv_rate="mean").sort_values("conv_rate", ascending=False)
display_tbl = src_tbl.assign(conv_rate=(src_tbl["conv_rate"] * 100).round(1))
print(display_tbl)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(src_tbl.index, src_tbl["leads"], color="#1F4E79"); axes[0].set_title("Lead volume by source")
axes[1].bar(src_tbl.index, src_tbl["conv_rate"] * 100, color="#2E8B57"); axes[1].set_title("Conversion rate (%) by source")
axes[1].axhline(df["converted"].mean() * 100, color="#C0392B", ls="--")
for ax in axes: plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
plt.show()
""")
md("### 4.2 Budget (customer-stated) distribution")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["budget_lakh"], bins=35, color="#1F4E79", ax=axes[0]); axes[0].set_title("Stated budget (INR lakh)")
sns.boxplot(data=df, x="property_type", y="budget_lakh", color="#EAF1F8", ax=axes[1]); axes[1].set_title("Budget by property type")
plt.setp(axes[1].get_xticklabels(), rotation=15)
plt.show()
""")
md("### 4.3 City and property-type patterns")
code("""
pt = df.pivot_table(index="city", columns="property_type", values="converted", aggfunc="mean") * 100
plt.figure(figsize=(8, 5)); sns.heatmap(pt, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Conversion rate (%) by city x property type (small cells are noisy)"); plt.show()
""")
md("### 4.4 Engagement, response time and readiness vs conversion")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
resp = df.groupby("response_band", observed=True)["converted"].mean() * 100
axes[0].bar(resp.index.astype(str), resp.values, color="#E07B39"); axes[0].set_title("By response-time band (%)")
sv = df.groupby("site_visit_completed")["converted"].mean() * 100
axes[1].bar(["No site visit", "Site visit"], sv.values, color=["#7F8C8D", "#2E8B57"]); axes[1].set_title("By site visit (%)")
ln = df.groupby("loan_preapproved")["converted"].mean() * 100
axes[2].bar(["No pre-approval", "Pre-approved"], ln.values, color=["#7F8C8D", "#2E8B57"]); axes[2].set_title("By loan pre-approval (%)")
plt.show()
""")
code("""
plt.figure(figsize=(9, 7))
sns.heatmap(df[config.NUMERIC_FEATURES + ["converted"]].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation matrix"); plt.show()
""")

md("""
## 5. Feature engineering
New columns: `total_engagement` (visits + 2×calls + WhatsApp), `response_band`, `followup_intensity`, `budget_band`,
`readiness_level` (loan pre-approved + site visit done).
""")
code("""
df[["lead_id", "total_engagement", "response_band", "followup_intensity", "budget_band", "readiness_level"]].head(8)
""")

md("""
## 6. Lead RFM-style (Recency / Frequency / Engagement) analysis
This is a **lead-specific adaptation** of RFM, not classic purchase RFM:
**R** = days since last contact (fewer is better), **F** = follow-ups made, **E** = weighted engagement.
Each is scored 1-5 by quintile; the sum (3-15) maps to Cold / Cool / Warm / Hot.
""")
code("""
rfe = df.groupby("rfe_segment", observed=True).agg(leads=("converted", "size"), conv_rate=("converted", "mean"),
                                                   avg_recency_days=("days_since_last_contact", "mean"),
                                                   avg_followups=("followup_count", "mean"),
                                                   avg_engagement=("total_engagement", "mean")).round(2)
print(rfe)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(rfe.index.astype(str), rfe["conv_rate"] * 100, color=["#7F8C8D", "#E0A339", "#E07B39", "#C0392B"])
ax.set_title("Conversion rate (%) by RFE segment"); plt.show()
""")

md("""
## 7. Explainable lead-priority score
`score = 30·engagement + 20·site_visit + 15·recency + 15·responsiveness + 10·loan_ready + 10·follow-up` (each scaled 0-1 with a documented cap).
Bands: **High ≥ 55**, **Medium ≥ 40**, **Low < 40**. No sensitive personal attributes are used.
""")
code("""
df = scoring.add_priority_score(df)
band = df.groupby("priority_band", observed=True).agg(leads=("converted", "size"), conv_rate=("converted", "mean")).reindex(config.BAND_ORDER)
print(band.round(3))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["priority_score"], bins=30, color="#1F4E79", ax=axes[0]); axes[0].set_title("Priority score distribution")
axes[1].bar(band.index.astype(str), band["conv_rate"] * 100, color=[config.BAND_COLORS[b] for b in band.index])
axes[1].set_title("Observed conversion rate (%) per band"); plt.show()
""")
code("""
# Explain one lead: which signals produced the score?
example = df.sort_values("priority_score", ascending=False).iloc[0]
print("Lead:", example["lead_id"], "| score:", example["priority_score"], "|", example["priority_band"])
scoring.explain_lead(example)
""")
code("""
df.sort_values("priority_score", ascending=False)[["lead_id", "lead_source", "city", "priority_score", "priority_band", "converted"]].head(10)
""")

md("""
## 8. Channel and revenue-potential analysis
Budget is the **customer-stated budget**, not company revenue. Spend is not in the data, so cost per conversion cannot be computed here.
""")
code("""
ch = visuals.channel_summary(df)
ch
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(ch.index, ch["leads"], color="#1F4E79"); axes[0].set_title("Leads")
axes[1].bar(ch.index, ch["conversion_rate"] * 100, color="#2E8B57"); axes[1].set_title("Conversion rate (%)")
axes[2].bar(ch.index, ch["total_stated_budget_cr"], color="#E07B39"); axes[2].set_title("Total stated budget (INR crore)")
for ax in axes: plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.show()
""")

md("""
## 9. Conversion model
**Leakage control:** `lead_status` (recorded after the outcome), `lead_id`, `campaign_name` and the target are excluded.
Models: majority-class baseline, Logistic Regression, Random Forest. Stratified 80/20 split + 5-fold CV on the training set.
""")
code("""
print("Features used:", model.FEATURES)
out = model.train_and_evaluate(df, save=False)
res = out["results"]
pd.DataFrame(res["models"]).T
""")
code("""
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name, (fpr, tpr) in out["curves"].items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC {res['models'][name]['roc_auc']:.2f})")
axes[0].plot([0, 1], [0, 1], "k:"); axes[0].legend(fontsize=7); axes[0].set_title("ROC curves (test)")
sns.heatmap(np.array(res["confusion_matrix_rf"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=["Pred No", "Pred Yes"], yticklabels=["Actual No", "Actual Yes"]); axes[1].set_title("Random Forest confusion matrix")
imp = out["importance"].head(10).iloc[::-1]
axes[2].barh(imp["feature"], imp["importance"], color="#1F4E79"); axes[2].set_title("Permutation importance")
plt.tight_layout(); plt.show()
""")
code("""
lift = out["lift"].round(3)
print(f"Top-decile lift: {res['top_decile_lift']}x | Top-3-decile capture of conversions: {res['top3_deciles_capture']:.0%}")
lift
""")

md("""
## 10. Business recommendations (from the output above)
The full, auto-generated write-up is in `reports/BUSINESS_RECOMMENDATIONS.md`. In short:
1. Order follow-ups by priority band (High first) and allow staff to override.
2. Track and shorten first-response time; test it with a proper SLA experiment before claiming causation.
3. Push engaged leads towards a site visit and loan pre-approval conversations.
4. Review channels using *real* spend data; the weakest converting source needs diagnosis before budget is cut.
5. Collect timestamps, spend and a precise outcome definition in the real CRM.
""")

md("""
## 11. Limitations and ethics
- **Synthetic data**: relationships were designed by the author; model metrics say nothing about the real market.
- **Correlation ≠ causation**: e.g. faster response is *associated* with conversion here; this does not prove it causes it.
- **Small hold-out set (240 leads)**: metric differences between models are noisy.
- **Possible leakage in real data**: fields recorded after the outcome must be excluded.
- **Privacy**: never upload names, phone numbers, e-mails or exact addresses; do not score using protected attributes.
- Scores are **estimates for human review**, not guarantees.
""")


# --------------------------------------------------------------------------- #
# Tiny executor
# --------------------------------------------------------------------------- #
def _split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def run_cell(src: str, ns: dict, count: int) -> list[dict]:
    outputs: list[dict] = []
    buf = io.StringIO()
    tree = ast.parse(src)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)
        value = eval(compile(last_expr, "<cell>", "eval"), ns) if last_expr is not None else None
    text = buf.getvalue()
    if text:
        outputs.append({"output_type": "stream", "name": "stdout", "text": _split_lines(text)})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                 "text/plain": ["<Figure>"]}})
    plt.close("all")
    if value is not None:
        data = {"text/plain": _split_lines(repr(value))}
        if hasattr(value, "_repr_html_"):
            data["text/html"] = _split_lines(value._repr_html_())
        outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
    return outputs


def build() -> None:
    os.chdir(ROOT / "notebooks")
    sys.path.insert(0, str(ROOT))
    warnings.filterwarnings("ignore")
    ns: dict = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, text in cells:
        if kind == MD:
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _split_lines(text)})
        else:
            count += 1
            outs = run_cell(text, ns, count)
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": count,
                             "outputs": outs, "source": _split_lines(text)})
    nb = {
        "cells": nb_cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {NB_PATH} ({len(nb_cells)} cells, {NB_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
