# End-to-End Requirements Specification - PropLead AI

**Project:** Real Estate Lead Conversion Intelligence System
**Programme:** PGDM - Marketing + Business Analytics
**Document status:** Prototype requirements (synthetic data). Source: `docs/PropLead_AI_Project_Proposal.docx`.

---

## 1. Business context

| Item | Description |
|---|---|
| Domain | Real-estate sales and digital marketing |
| Problem | Leads arrive from portals, Google Ads, Instagram, website, referrals and walk-ins. Teams often treat all leads alike, causing delayed follow-ups, inefficient marketing spend and missed sales. |
| Proposed solution | A data-driven **decision-support** system that (a) cleans and analyses lead data, (b) gives an explainable follow-up priority score, (c) estimates conversion likelihood, (d) compares channels, and (e) presents everything in a dashboard. |
| Out of scope | Guaranteeing sales, replacing sales judgement, automated customer messaging, live CRM integration, using personal identifiers. |

### 1.1 Stakeholders and users
| Stakeholder | Need |
|---|---|
| Sales executive | Which leads should I call first today, and why? |
| Sales manager | Are follow-ups prioritised sensibly? How is the team performing by lead band? |
| Marketing manager | Which channels bring leads that actually convert? Where should the budget be reviewed? |
| Data analyst / student | Reproducible pipeline, documented assumptions, clear limitations. |
| Faculty / evaluator | Evidence of end-to-end analytics skills with ethical safeguards. |

### 1.2 Business questions to answer
1. Which sources, cities, budgets and property types show higher conversion?
2. Which engagement and follow-up behaviours are associated with conversion?
3. Which leads deserve attention first, and can the reasoning be explained to a salesperson?
4. How well can conversion likelihood be estimated from available fields?
5. Which channels merit a budget review or a test?

---

## 2. Objectives and traceability

| # | Objective (from proposal) | Requirement IDs | Where it lives in the repo |
|---|---|---|---|
| O1 | Analyse lead quality and conversion across channels, locations, budgets, property types | FR-04, FR-08 | `notebooks/`, `images/02-07`, `src/visuals.py` |
| O2 | Clean and prepare data; check missing, duplicates, inconsistent entries | FR-02, FR-03 | `src/data_cleaning.py`, `reports/cleaning_report.json` |
| O3 | Engagement and follow-up features | FR-05, FR-06 | `src/features.py` |
| O4 | Explainable lead-priority score | FR-07 | `src/scoring.py`, `images/diagram_scoring_logic.png` |
| O5 | Classification model for conversion likelihood | FR-09, FR-10 | `src/model.py`, `reports/model_metrics.json` |
| O6 | Dashboard (volume, conversion, channel, high-priority leads) | FR-11 | `app.py` |
| O7 | Recommendations for follow-up and marketing budget review | FR-12 | `reports/BUSINESS_RECOMMENDATIONS.md` |
| O8 | Limitations, privacy safeguards, next steps | NFR-04..06 | `docs/LIMITATIONS_AND_ETHICS.md` |

---

## 3. Functional requirements

### 3.1 Data
| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-01 | Generate a reproducible **synthetic** dataset of 1,200 leads with the field groups in the data dictionary. | `python -m src.generate_dataset` outputs 1,200 rows x 18 columns; same seed gives identical file. |
| FR-02 | Profile data quality: missing values, duplicates, inconsistent category spellings, invalid ranges. | `reports/cleaning_report.json` lists counts for each check. |
| FR-03 | Clean data: standardise categories, remove duplicates, validate binary/count fields, impute missing values (flagged), cap extreme outliers. | After cleaning: 0 missing values, 6 canonical lead sources, `*_imputed` flags present. |

### 3.2 Analysis and features
| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-04 | EDA on lead volume, conversion rate, source mix, budget distribution, city and property-type patterns. | Charts `02`-`08` generated; notebook sections 4.1-4.4 populated. |
| FR-05 | Engineer features: `total_engagement`, `response_band`, `followup_intensity`, `budget_band`, `readiness_level`. | Columns exist and have no nulls. |
| FR-06 | Lead RFM-style analysis: Recency, Frequency, Engagement scores (1-5), total (3-15), segment Cold/Cool/Warm/Hot. | `R_score`, `F_score`, `E_score`, `RFE_total`, `rfe_segment` produced; documented as a lead-specific adaptation. |
| FR-07 | Transparent priority score 0-100 with Low/Medium/High bands, per-component breakdown, **no sensitive attributes**. | Weights sum to 100; `explain_lead()` returns points per component; test `test_priority_score_properties` passes. |
| FR-08 | Channel analysis: leads, conversions, conversion rate, median/total **customer-stated** budget, high-priority share. | `reports/channel_summary.csv`; budget never labelled as revenue. |

### 3.3 Modelling
| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-09 | Train a baseline and a Random Forest classifier (plus Logistic Regression) with stratified split and cross-validation. | `reports/model_metrics.json` has precision, recall, F1, ROC-AUC, PR-AUC, CV AUC for each model. |
| FR-10 | Prevent leakage: exclude `lead_status`, `lead_id`, `converted`, `campaign_name`; show out-of-fold probabilities to users. | Test `test_model_has_no_leakage_columns` passes. |

### 3.4 Dashboard and reporting
| ID | Requirement | Acceptance criterion |
|---|---|---|
| FR-11 | Streamlit dashboard with sidebar filters (source, city, property type, budget) and five tabs: Overview, Channels, Lead priority, Model, Score a lead. | `streamlit run app.py` starts; KPIs update with filters; priority list downloadable as CSV; per-lead score breakdown visible. |
| FR-12 | Auto-generate a business-recommendation summary from actual outputs, worded as association (not causation) with a synthetic-data warning. | `reports/BUSINESS_RECOMMENDATIONS.md` regenerated by `run_pipeline.py`. |
| FR-13 | Provide diagrams: workflow, architecture, data model, scoring logic, dashboard wireframe. | Files `images/diagram_*.png`. |
| FR-14 | One-command reproduction of every artefact. | `python run_pipeline.py` completes without error. |

---

## 4. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Reproducibility | Fixed random seed (42); pinned behaviour via `src/config.py`. |
| NFR-02 | Performance | Full pipeline < 1 minute on a normal laptop; dashboard loads < 5 s after first run. |
| NFR-03 | Maintainability | Modular `src/` package, docstrings, tests in `tests/`, all tunables in `config.py`. |
| NFR-04 | Privacy | No names, phone numbers, e-mails or exact addresses anywhere in the repo; `.gitignore` blocks real-data paths. |
| NFR-05 | Fairness | Protected/sensitive personal attributes are never used in scoring or modelling. |
| NFR-06 | Transparency | Scores presented as estimates; human review and override encouraged; limitations documented. |
| NFR-07 | Portability | Python 3.10+; runs on Windows, macOS, Linux; dependencies in `requirements.txt`. |
| NFR-08 | Explainability | Rule-based score decomposed into component points; permutation importance for the model. |

---

## 5. Data requirements

**Prototype:** synthetic data only (generated by `src/generate_dataset.py`).
**Real-world version - requirements before use:**
1. Written authorisation from the data owner; consent where required by law (e.g. India's DPDP Act, 2023).
2. Direct identifiers removed before analysis (hash or drop `lead_id` links to the CRM).
3. A clearly defined outcome (site visit booked / qualified / completed sale) and an observation window.
4. Event timestamps so features contain only information available *before* the outcome.
5. Optional: marketing spend per channel/campaign for cost-per-conversion analysis.
6. Minimum volume: several hundred positive outcomes for a stable classifier.

Full field list: [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

---

## 6. Technology requirements

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.10+ | Everything |
| Data | pandas, NumPy | Cleaning, features |
| ML | scikit-learn, joblib | Models, persistence |
| Charts | Matplotlib, Seaborn, Plotly | Static and interactive visuals |
| App | Streamlit | Dashboard |
| Notebook | Jupyter | EDA walkthrough |
| Version control | Git + GitHub | Portfolio hosting |
| Optional extensions | Excel, SQL, Power BI | Analyst hand-off, database storage, BI dashboard |

**Setup:** see `README.md` (virtual environment, `pip install -r requirements.txt`, `python run_pipeline.py`, `streamlit run app.py`).

---

## 7. Deliverables checklist

- [x] Documented synthetic dataset (`data/sample_real_estate_leads.csv`, `docs/DATA_DICTIONARY.md`)
- [x] Jupyter notebook with EDA + feature engineering (`notebooks/PropLead_AI_Analysis.ipynb`)
- [x] Streamlit dashboard (`app.py`)
- [x] GitHub README with setup, fields, workflow, limitations, future work
- [x] Business recommendation summary from actual output (`reports/BUSINESS_RECOMMENDATIONS.md`)
- [x] Diagrams, charts and images (`images/`)
- [x] Source code modules and tests (`src/`, `tests/`)
- [x] Ethics and limitations document

---

## 8. Suggested project plan (6 weeks)

| Week | Activities | Output |
|---|---|---|
| 1 | Problem framing, literature/context study, finalise requirements | This document |
| 2 | Dataset design/generation, cleaning, data dictionary | Clean dataset, cleaning report |
| 3 | EDA, feature engineering, RFE analysis | Notebook sections 4-6 |
| 4 | Priority score design, channel analysis | `scoring.py`, channel report |
| 5 | Modelling, evaluation, leakage checks, dashboard | `model.py`, `app.py` |
| 6 | Recommendations, documentation, GitHub release, presentation | README, report, final repo |

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic data mistaken for real evidence | Misleading conclusions | Warnings in README, notebook, report and dashboard |
| Target leakage in real CRM data | Over-optimistic model | Leakage checklist, timestamped features, out-of-fold scoring |
| Biased scoring | Unfair treatment of leads | No sensitive attributes; periodic review with managers |
| Small test set | Noisy metrics | Cross-validation reported alongside hold-out |
| Over-reliance on score | Missed good leads | Human override; scores framed as estimates |
| Privacy breach on GitHub | Legal and ethical harm | `.gitignore`, no PII, anonymised data only |

---

## 10. Glossary
- **Lead:** a prospective buyer who has shown interest.
- **Conversion:** the defined positive outcome (`converted = 1`; in the demo, a booking).
- **RFE:** Recency-Frequency-Engagement, a lead-specific adaptation of RFM.
- **Out-of-fold probability:** a prediction for a lead made by a model that did not train on that lead.
- **Lift:** conversion rate of a group divided by the overall rate.
- **Target leakage:** using information unavailable at prediction time (e.g. the final status) as a feature.
