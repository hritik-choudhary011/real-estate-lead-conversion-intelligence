"""Build reports/BUSINESS_RECOMMENDATIONS.md from the ACTUAL pipeline output.

Every number below is computed from the data at run time. The text uses
cautious wording: observed association, not causation. Because the demo
dataset is synthetic, the report is a template for what real results would look like.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.visuals import channel_summary


def _pct(x: float) -> str:
    return f"{x:.1%}"


def build_report(df: pd.DataFrame, metrics: dict, lift: pd.DataFrame, importance: pd.DataFrame) -> str:
    n, overall = len(df), df["converted"].mean()
    ch = channel_summary(df)
    best, worst = ch.index[0], ch.index[-1]
    band = df.groupby("priority_band", observed=True).agg(leads=("converted", "size"), rate=("converted", "mean"))
    resp = df.groupby("response_band", observed=True)["converted"].mean()
    sv = df.groupby("site_visit_completed")["converted"].mean()
    loan = df.groupby("loan_preapproved")["converted"].mean()
    rf, lr = metrics["models"]["Random Forest"], metrics["models"]["Logistic Regression"]
    base = metrics["models"]["Baseline (majority class)"]
    top3 = metrics["top3_deciles_capture"]

    lines = [
        "# Business Recommendations (data-driven summary)",
        "",
        "> **Read this first:** the dataset is **synthetic** (generated for demonstration). The relationships below were "
        "written into the generator, so these findings show *how the workflow reports results* - they are **not evidence "
        "of real real-estate market behaviour**. Re-run the pipeline on authorised, anonymised CRM data before acting.",
        "",
        "## 1. Headline numbers",
        f"- Leads analysed: **{n:,}**; converted: **{int(df['converted'].sum()):,}** (**{_pct(overall)}**).",
        f"- Best-converting source: **{best}** ({_pct(ch.loc[best, 'conversion_rate'])}); "
        f"lowest: **{worst}** ({_pct(ch.loc[worst, 'conversion_rate'])}).",
        f"- Priority bands (observed conversion): High **{_pct(band.loc['High','rate'])}** "
        f"({int(band.loc['High','leads'])} leads), Medium **{_pct(band.loc['Medium','rate'])}** "
        f"({int(band.loc['Medium','leads'])}), Low **{_pct(band.loc['Low','rate'])}** ({int(band.loc['Low','leads'])}).",
        "",
        "## 2. Recommendations for sales follow-up",
        f"1. **Work the High band first.** High-priority leads converted at {_pct(band.loc['High','rate'])} versus "
        f"{_pct(band.loc['Low','rate'])} for Low - roughly {band.loc['High','rate'] / max(band.loc['Low','rate'], 1e-9):.1f}x. "
        "Use the score to order the daily call list; let staff override it using local knowledge.",
        f"2. **Respond fast.** Leads answered within 2 hours converted at {_pct(resp.get('0-2h', float('nan')))} versus "
        f"{_pct(resp.get('24h+', float('nan')))} when the first response took over 24 hours (association only; "
        "test with a controlled response-time SLA before claiming impact).",
        f"3. **Push for a site visit.** Leads with a completed site visit converted at {_pct(sv[1])} versus {_pct(sv[0])} without one.",
        f"4. **Surface finance readiness early.** Loan pre-approved leads converted at {_pct(loan[1])} versus {_pct(loan[0])}; "
        "offer loan-assistance conversations to leads that are engaged but not pre-approved.",
        f"5. **Model-assisted calling list.** Contacting the top 30% of leads by predicted probability would have reached about "
        f"**{_pct(top3)}** of all conversions in out-of-fold scoring (top-decile lift {metrics['top_decile_lift']}x).",
        "",
        "## 3. Recommendations for marketing budget review",
        "| Source | Leads | Conv. rate | Median stated budget (lakh) | High-priority share |",
        "|---|---:|---:|---:|---:|",
    ]
    for src, r in ch.iterrows():
        lines.append(f"| {src} | {int(r['leads'])} | {_pct(r['conversion_rate'])} | {r['median_budget_lakh']:.0f} | {_pct(r['high_priority_share'])} |")
    lines += [
        "",
        f"- **Protect and test** high-converting, low-volume sources ({', '.join(ch.index[:2])}): pilot small budget increases and measure whether volume holds quality.",
        f"- **Review** the weakest source ({worst}): check lead-form quality, targeting and follow-up speed before cutting spend. "
        "Cost data is not part of the dataset; add real spend per channel to compute cost per conversion (the dashboard has an input for illustrative costs).",
        "- Budget values are **customer-stated budgets**, not company revenue.",
        "",
        "## 4. Model summary",
        "| Model | Precision | Recall | F1 | ROC-AUC (test) | CV ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics["models"].items():
        lines.append(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['roc_auc']:.2f} | "
                     f"{m['cv_roc_auc_mean']:.2f} +/- {m['cv_roc_auc_std']:.2f} |")
    lines += [
        "",
        f"- Hold-out set is only {metrics['n_test']} leads, so differences between Logistic Regression "
        f"(AUC {lr['roc_auc']:.2f}) and Random Forest (AUC {rf['roc_auc']:.2f}) are within noise; "
        f"both clearly beat the baseline (AUC {base['roc_auc']:.2f}). Cross-validated scores are the steadier comparison.",
        f"- Most influential features (permutation importance): {', '.join(importance['feature'].head(4))}.",
        "- Scores are **estimates**; they support, not replace, the judgement of sales staff.",
        "",
        "## 5. Data improvements before real deployment",
        "1. Replace synthetic data with consented, anonymised CRM exports (no names, phones, e-mails or exact addresses).",
        "2. Record the timestamp of each event so features use only information known *before* the outcome.",
        "3. Add marketing spend per channel/campaign to evaluate cost per lead and cost per booking.",
        "4. Define the outcome precisely (site visit booked vs. qualified vs. sale) and keep it consistent.",
        "5. Re-check score weights, thresholds and fairness with sales managers; monitor drift monthly.",
        "",
        "_Generated automatically by `python run_pipeline.py`._",
    ]
    return "\n".join(lines)


def save_report(*args) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "BUSINESS_RECOMMENDATIONS.md").write_text(build_report(*args), encoding="utf-8")
