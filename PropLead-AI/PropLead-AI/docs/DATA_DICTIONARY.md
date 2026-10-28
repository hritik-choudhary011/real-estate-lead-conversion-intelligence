# Data Dictionary

File: `data/sample_real_estate_leads.csv` - **1,200 synthetic records, 18 columns.** No real customer data.
Deliberate data-quality issues (missing values, inconsistent spellings) are included so the cleaning step has work to do.

| Column | Group | Type | Description / allowed values | Used in model? |
|---|---|---|---|---|
| `lead_id` | Identity | text | Demo ID `L0001`-`L1200` (non-personal) | No (identifier) |
| `lead_source` | Acquisition | category | Property Portal, Google Ads, Instagram, Website, Referral, Walk-in | Yes |
| `campaign_name` | Acquisition | category | Campaign inside the source (e.g. `GAds_Search_Brand`) | No (nested in source) |
| `city` | Acquisition | category | Noida, Gurugram, Ghaziabad, Lucknow, Jaipur, Pune, Bengaluru, Hyderabad | Yes |
| `property_type` | Requirement | category | Apartment, Builder Floor, Villa, Plot, Commercial | Yes |
| `preferred_bhk` | Requirement | int | 1-5; `0` = not applicable (plot/commercial) | Yes |
| `budget_lakh` | Requirement | float | Customer-**stated** budget in INR lakh (not company revenue) | Yes |
| `website_visits` | Engagement | int | Number of website visits | Yes |
| `calls_connected` | Engagement | int | Calls that connected | Yes |
| `whatsapp_replies` | Engagement | int | WhatsApp replies from the lead | Yes |
| `site_visit_completed` | Sales activity | 0/1 | Site visit completed | Yes |
| `followup_count` | Sales activity | int | Follow-ups made by sales team (>=1) | Yes |
| `response_time_hours` | Sales activity | float | Hours until first response | Yes |
| `days_since_last_contact` | Sales activity | int | Recency in days | Yes |
| `loan_preapproved` | Readiness | 0/1 | Loan pre-approval available | Yes |
| `discount_interest` | Readiness | 0/1 | Asked about discounts/offers | Yes |
| `lead_status` | Outcome | category | Booked, Lost, Nurturing, Negotiation, New / Uncontacted | **No - leakage** (recorded after outcome) |
| `converted` | Target | 0/1 | 1 = booked / converted | Target |

## Derived columns (created by the pipeline)
| Column | Definition |
|---|---|
| `response_time_imputed`, `budget_imputed` | 1 if the value was missing and imputed |
| `total_engagement` | `website_visits + 2 x calls_connected + whatsapp_replies` |
| `interactions_total` | calls + WhatsApp replies + follow-ups |
| `response_band` | 0-2h, 2-6h, 6-24h, 24h+ |
| `followup_intensity` | Low (1-2), Medium (3-5), High (6+) |
| `budget_band` | <50L, 50-100L, 100-200L, 200L+ |
| `readiness_level` | `loan_preapproved + site_visit_completed` (0-2) |
| `R_score`, `F_score`, `E_score` | Quintile scores 1-5 (5 = best) |
| `RFE_total`, `rfe_segment` | Sum 3-15; Cold (<=5), Cool (6-8), Warm (9-11), Hot (12+) |
| `priority_score`, `priority_band`, `priority_rank` | Explainable 0-100 score; High >= 55, Medium >= 40, Low < 40 |
| `conversion_probability`, `probability_decile` | Random-Forest out-of-fold probability; decile 1 = highest |

## Fields that must NEVER be added to the public repo
Customer name, phone number, e-mail, exact address, Aadhaar/PAN or any government ID, and any
protected attribute (religion, caste, gender, etc.) used for scoring.
