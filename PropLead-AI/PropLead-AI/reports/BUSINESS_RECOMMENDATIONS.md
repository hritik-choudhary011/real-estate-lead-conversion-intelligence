# Business Recommendations (data-driven summary)

> **Read this first:** the dataset is **synthetic** (generated for demonstration). The relationships below were written into the generator, so these findings show *how the workflow reports results* - they are **not evidence of real real-estate market behaviour**. Re-run the pipeline on authorised, anonymised CRM data before acting.

## 1. Headline numbers
- Leads analysed: **1,200**; converted: **258** (**21.5%**).
- Best-converting source: **Walk-in** (34.6%); lowest: **Property Portal** (17.5%).
- Priority bands (observed conversion): High **42.3%** (381 leads), Medium **16.7%** (443), Low **6.1%** (376).

## 2. Recommendations for sales follow-up
1. **Work the High band first.** High-priority leads converted at 42.3% versus 6.1% for Low - roughly 6.9x. Use the score to order the daily call list; let staff override it using local knowledge.
2. **Respond fast.** Leads answered within 2 hours converted at 35.1% versus 13.3% when the first response took over 24 hours (association only; test with a controlled response-time SLA before claiming impact).
3. **Push for a site visit.** Leads with a completed site visit converted at 41.1% versus 13.7% without one.
4. **Surface finance readiness early.** Loan pre-approved leads converted at 33.2% versus 13.9%; offer loan-assistance conversations to leads that are engaged but not pre-approved.
5. **Model-assisted calling list.** Contacting the top 30% of leads by predicted probability would have reached about **60.9%** of all conversions in out-of-fold scoring (top-decile lift 2.98x).

## 3. Recommendations for marketing budget review
| Source | Leads | Conv. rate | Median stated budget (lakh) | High-priority share |
|---|---:|---:|---:|---:|
| Walk-in | 107 | 34.6% | 82 | 27.1% |
| Referral | 133 | 33.1% | 95 | 34.6% |
| Website | 173 | 20.2% | 92 | 31.8% |
| Instagram | 185 | 18.9% | 97 | 34.1% |
| Google Ads | 264 | 18.2% | 93 | 31.4% |
| Property Portal | 338 | 17.5% | 96 | 31.1% |

- **Protect and test** high-converting, low-volume sources (Walk-in, Referral): pilot small budget increases and measure whether volume holds quality.
- **Review** the weakest source (Property Portal): check lead-form quality, targeting and follow-up speed before cutting spend. Cost data is not part of the dataset; add real spend per channel to compute cost per conversion (the dashboard has an input for illustrative costs).
- Budget values are **customer-stated budgets**, not company revenue.

## 4. Model summary
| Model | Precision | Recall | F1 | ROC-AUC (test) | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (majority class) | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 +/- 0.00 |
| Logistic Regression | 0.44 | 0.60 | 0.50 | 0.74 | 0.78 +/- 0.03 |
| Random Forest | 0.49 | 0.46 | 0.48 | 0.72 | 0.78 +/- 0.03 |

- Hold-out set is only 240 leads, so differences between Logistic Regression (AUC 0.74) and Random Forest (AUC 0.72) are within noise; both clearly beat the baseline (AUC 0.50). Cross-validated scores are the steadier comparison.
- Most influential features (permutation importance): loan_preapproved, site_visit_completed, total_engagement, lead_source.
- Scores are **estimates**; they support, not replace, the judgement of sales staff.

## 5. Data improvements before real deployment
1. Replace synthetic data with consented, anonymised CRM exports (no names, phones, e-mails or exact addresses).
2. Record the timestamp of each event so features use only information known *before* the outcome.
3. Add marketing spend per channel/campaign to evaluate cost per lead and cost per booking.
4. Define the outcome precisely (site visit booked vs. qualified vs. sale) and keep it consistent.
5. Re-check score weights, thresholds and fairness with sales managers; monitor drift monthly.

_Generated automatically by `python run_pipeline.py`._