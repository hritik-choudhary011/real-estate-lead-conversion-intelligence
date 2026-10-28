"""PropLead AI - Streamlit dashboard.

Run:  streamlit run app.py

Decision support only: scores are estimates built on SYNTHETIC demo data.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, model, scoring
from src.visuals import channel_summary

st.set_page_config(page_title="PropLead AI", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (runs the pipeline once if artefacts are missing)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Preparing data and models (first run only) ...")
def load_scored() -> pd.DataFrame:
    if not (config.SCORED_PATH.exists() and config.METRICS_PATH.exists() and config.MODEL_PATH.exists()):
        from run_pipeline import run_all
        run_all(make_charts=False, verbose=False)
    return pd.read_csv(config.SCORED_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text())


@st.cache_resource
def load_rf():
    return model.load_model()


df_all = load_scored()
metrics = load_metrics()
band_order = config.BAND_ORDER

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🏠 PropLead AI")
st.sidebar.caption("Real Estate Lead Conversion Intelligence")
sources = st.sidebar.multiselect("Lead source", sorted(df_all["lead_source"].unique()),
                                 default=sorted(df_all["lead_source"].unique()))
cities = st.sidebar.multiselect("City", sorted(df_all["city"].unique()),
                                default=sorted(df_all["city"].unique()))
ptypes = st.sidebar.multiselect("Property type", sorted(df_all["property_type"].unique()),
                                default=sorted(df_all["property_type"].unique()))
b_lo, b_hi = float(df_all["budget_lakh"].min()), float(df_all["budget_lakh"].max())
budget = st.sidebar.slider("Stated budget (INR lakh)", b_lo, b_hi, (b_lo, b_hi))

df = df_all[
    df_all["lead_source"].isin(sources) & df_all["city"].isin(cities)
    & df_all["property_type"].isin(ptypes) & df_all["budget_lakh"].between(*budget)
]

st.sidebar.markdown("---")
st.sidebar.info("**Synthetic demo data.** Scores are estimates to support - not replace - sales judgement.")

st.title("PropLead AI - Lead Conversion Intelligence")
if df.empty:
    st.warning("No leads match the current filters.")
    st.stop()

tab_over, tab_ch, tab_pri, tab_model, tab_new = st.tabs(
    ["📊 Overview", "📣 Channels", "🎯 Lead priority", "🤖 Model", "🧮 Score a lead"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total leads", f"{len(df):,}")
    c2.metric("Converted", f"{int(df['converted'].sum()):,}")
    c3.metric("Conversion rate", f"{df['converted'].mean():.1%}")
    c4.metric("High-priority leads", f"{int((df['priority_band'] == 'High').sum()):,}")

    left, right = st.columns(2)
    by_band = (df.groupby("priority_band")["converted"].agg(leads="size", conversion_rate="mean")
               .reindex(band_order).reset_index())
    fig = px.bar(by_band, x="priority_band", y="leads", color="priority_band",
                 color_discrete_map=config.BAND_COLORS, title="Leads per priority band",
                 category_orders={"priority_band": band_order})
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(by_band, x="priority_band", y="conversion_rate", color="priority_band",
                 color_discrete_map=config.BAND_COLORS, title="Observed conversion rate per band",
                 category_orders={"priority_band": band_order})
    fig.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig, use_container_width=True)

    city_t = df.groupby("city")["converted"].agg(leads="size", conversion_rate="mean").reset_index()
    fig = px.bar(city_t.sort_values("leads", ascending=False), x="city", y="leads",
                 color="conversion_rate", color_continuous_scale="Blues",
                 title="Leads by city (colour = conversion rate)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
with tab_ch:
    ch = channel_summary(df).reset_index()
    a, b = st.columns(2)
    fig = px.bar(ch, x="lead_source", y="leads", title="Lead volume by source", color_discrete_sequence=["#1F4E79"])
    a.plotly_chart(fig, use_container_width=True)
    fig = px.bar(ch.sort_values("conversion_rate", ascending=False), x="lead_source", y="conversion_rate",
                 title="Conversion rate by source", color_discrete_sequence=["#2E8B57"])
    fig.update_yaxes(tickformat=".0%")
    b.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel table")
    st.dataframe(ch.rename(columns={
        "lead_source": "Source", "leads": "Leads", "conversions": "Conversions",
        "conversion_rate": "Conv. rate", "median_budget_lakh": "Median stated budget (lakh)",
        "total_stated_budget_cr": "Total stated budget (cr)", "avg_priority_score": "Avg priority",
        "high_priority_share": "High-priority share", "avg_response_hours": "Avg response (h)"}),
        use_container_width=True, hide_index=True)
    st.caption("Budgets are customer-stated, not company revenue.")

    st.subheader("Cost per conversion (illustrative)")
    st.caption("The dataset has no spend data. Enter ASSUMED cost per lead (INR) for each source to see how the "
               "ranking could change. These inputs are your assumptions, not facts.")
    cols = st.columns(len(ch))
    costs = {}
    for col, src in zip(cols, ch["lead_source"]):
        costs[src] = col.number_input(src, min_value=0, value=0, step=100, key=f"cost_{src}")
    if any(costs.values()):
        ch["assumed_spend"] = ch["lead_source"].map(costs) * ch["leads"]
        ch["cost_per_conversion"] = ch["assumed_spend"] / ch["conversions"].replace(0, pd.NA)
        st.dataframe(ch[["lead_source", "leads", "conversions", "assumed_spend", "cost_per_conversion"]],
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Lead priority
# --------------------------------------------------------------------------- #
with tab_pri:
    st.markdown("**Explainable priority score (0-100):** engagement 30 + site visit 20 + recency 15 + "
                "responsiveness 15 + loan pre-approved 10 + follow-ups 10. "
                f"Bands: High >= {config.BAND_HIGH_MIN:.0f}, Medium >= {config.BAND_MEDIUM_MIN:.0f}, else Low.")
    f1, f2, f3 = st.columns(3)
    bands = f1.multiselect("Priority band", band_order, default=["High"])
    only_open = f2.checkbox("Hide already-converted leads", value=True,
                            help="'converted' is the known outcome in this demo; in real use you would score only open leads.")
    top_n = f3.slider("Rows", 10, 200, 25)
    view = df[df["priority_band"].isin(bands)]
    if only_open:
        view = view[view["converted"] == 0]
    view = view.sort_values("priority_score", ascending=False).head(top_n)
    show_cols = ["lead_id", "lead_source", "city", "property_type", "budget_lakh", "priority_score",
                 "priority_band", "conversion_probability", "site_visit_completed", "loan_preapproved",
                 "days_since_last_contact", "response_time_hours", "followup_count"]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True,
                 column_config={
                     "priority_score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=100, format="%.1f"),
                     "conversion_probability": st.column_config.ProgressColumn("Model probability (out-of-fold)", min_value=0, max_value=1, format="%.2f"),
                 })
    st.download_button("⬇️ Download this list (CSV)", view[show_cols].to_csv(index=False).encode(),
                       "priority_leads.csv", "text/csv")

    if not view.empty:
        st.subheader("Why this score? (breakdown for one lead)")
        pick = st.selectbox("Lead", view["lead_id"].tolist())
        row = df_all[df_all["lead_id"] == pick].iloc[0]
        exp = scoring.explain_lead(row)
        fig = px.bar(exp.sort_values("points"), x="points", y="component", orientation="h",
                     text="points", title=f"{pick}: {row['priority_score']:.1f} points ({row['priority_band']})")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
with tab_model:
    st.markdown(f"Hold-out test set: **{metrics['n_test']}** leads - train: **{metrics['n_train']}**. "
                "Small test sets give noisy metrics; read the cross-validated column too.")
    mt = pd.DataFrame(metrics["models"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(mt, use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    cm = metrics["confusion_matrix_rf"]
    fig = px.imshow(cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                    color_continuous_scale="Blues", title="Random Forest confusion matrix")
    a.plotly_chart(fig, use_container_width=True)
    imp = pd.DataFrame(metrics["top_features"]).sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 title="Top features (permutation importance, drop in ROC-AUC)")
    b.plotly_chart(fig, use_container_width=True)

    lift = pd.read_csv(config.REPORTS_DIR / "lift_table.csv")
    fig = px.bar(lift, x="decile", y="conv_rate", title="Conversion rate by predicted-probability decile (1 = highest)")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Top-decile lift: {metrics['top_decile_lift']}x. Top 3 deciles capture "
               f"{metrics['top3_deciles_capture']:.0%} of conversions (out-of-fold).")

# --------------------------------------------------------------------------- #
# Score a new lead
# --------------------------------------------------------------------------- #
with tab_new:
    st.markdown("Enter details of a lead to see its priority score and the Random Forest's estimated "
                "probability. **Do not enter names, phone numbers or e-mail addresses** - none are needed.")
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("Lead source", sorted(df_all["lead_source"].unique()))
    city = c1.selectbox("City", sorted(df_all["city"].unique()))
    ptype = c1.selectbox("Property type", sorted(df_all["property_type"].unique()))
    bhk = c1.selectbox("Preferred BHK (0 = n/a)", [0, 1, 2, 3, 4, 5], index=3)
    bud = c2.number_input("Stated budget (lakh)", 10.0, 1000.0, 90.0, step=5.0)
    visits = c2.number_input("Website visits", 0, 50, 4)
    calls = c2.number_input("Connected calls", 0, 20, 1)
    wa = c2.number_input("WhatsApp replies", 0, 30, 2)
    fu = c3.number_input("Follow-ups made", 1, 30, 3)
    rt = c3.number_input("First response time (hours)", 0.1, 100.0, 4.0)
    days = c3.number_input("Days since last contact", 0, 365, 5)
    sv = c3.checkbox("Site visit completed")
    loan = c3.checkbox("Loan pre-approved")
    disc = c3.checkbox("Interested in discount")

    if st.button("Score this lead", type="primary"):
        new = {"lead_source": src, "city": city, "property_type": ptype, "preferred_bhk": bhk,
               "budget_lakh": bud, "website_visits": visits, "calls_connected": calls,
               "whatsapp_replies": wa, "followup_count": fu, "response_time_hours": rt,
               "days_since_last_contact": days, "site_visit_completed": int(sv),
               "loan_preapproved": int(loan), "discount_interest": int(disc)}
        new["total_engagement"] = sum(new[k] * w for k, w in config.ENGAGEMENT_WEIGHTS.items())
        frame = pd.DataFrame([new])
        comps = scoring.score_components(frame)
        score = float(comps.sum(axis=1).iloc[0])
        band = str(scoring.assign_band(pd.Series([score])).iloc[0])
        prob = model.predict_proba_row(load_rf(), new)
        m1, m2, m3 = st.columns(3)
        m1.metric("Priority score", f"{score:.1f} / 100")
        m2.metric("Priority band", band)
        m3.metric("Estimated conversion probability", f"{prob:.0%}")
        st.bar_chart(comps.iloc[0])
        st.caption("Estimates from a model trained on synthetic data - for demonstration only.")
