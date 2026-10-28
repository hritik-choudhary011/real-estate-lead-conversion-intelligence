# Limitations, Privacy and Ethics

## Limitations
1. **Synthetic data.** All 1,200 records are generated. Patterns (e.g. referrals converting better, faster responses converting better) were put there by the generator and must not be quoted as real-market findings.
2. **Correlation, not causation.** Observed associations (response time, site visits, loan pre-approval) do not prove that changing them would change outcomes. Use controlled tests (A/B follow-up SLAs) before drawing causal conclusions.
3. **Small sample / small test set.** 240 hold-out leads give wide confidence intervals; model rankings can flip with another seed.
4. **No cost data.** Channel ROI and cost per conversion cannot be computed from the dataset; the dashboard offers an illustrative input only.
5. **Budget is customer-stated.** It is not company revenue or deal value.
6. **Outcome definition.** The demo uses a single `converted` flag; a real CRM needs a precise definition (booked site visit, qualified lead, sale) and a fixed observation window.
7. **Possible leakage in real data.** Fields such as `site_visit_completed` or `followup_count` may be updated after the outcome in a real CRM; verify with event timestamps.
8. **Threshold and weight choices.** Score weights and band cut-offs are documented judgement calls.
9. **Model persistence.** `models/rf_conversion_model.joblib` is tied to the scikit-learn version used to create it; regenerate with `python run_pipeline.py` if loading fails.

## Privacy safeguards
- No personal identifiers (names, phone numbers, e-mails, exact addresses, government IDs) in the repository.
- `lead_id` values are synthetic; with real data, replace CRM IDs with random surrogate keys.
- `.gitignore` blocks `data/real/`, `data/private/`, `.env` and `*_real_*.csv`.
- Use authorised, anonymised data only; follow applicable law (for India, the Digital Personal Data Protection Act, 2023) and the data owner's policy.
- The dashboard's "Score a lead" form asks for no personal information.

## Fairness
- Sensitive or protected attributes (religion, caste, gender, nationality, disability, etc.) are never used.
- Location (`city`) is used as a market variable in the model only; check that it does not act as a proxy for protected groups before real use.
- Audit band distribution and conversion by segment regularly.

## Responsible use
- Scores are **estimates for human review**; sales staff can and should override them.
- Never use the score to deny service or to treat a lead unfairly; use it only to order follow-ups.
- Log the model/score version used for each decision so results can be audited.

## Next steps
1. Validate on authorised, anonymised CRM data and re-calibrate weights.
2. Add event timestamps, marketing spend and a precise outcome definition.
3. Move preprocessing fully inside the cross-validation pipeline; try gradient boosting and probability calibration.
4. Add SQL storage and a Power BI dashboard for non-technical users.
5. Run an A/B test on follow-up prioritisation to measure real impact.
6. Add drift monitoring and a feedback loop where sales staff correct scores.
