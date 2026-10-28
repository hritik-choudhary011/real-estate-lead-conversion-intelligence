"""Run the whole PropLead AI pipeline end to end.

    python -m src.run_pipeline

Outputs: cleaned + scored CSVs, model file, metrics JSON, charts/diagrams, business report.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, data_cleaning, features, generate_dataset, generate_report, model, scoring, visuals


def run_all(verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    if not config.RAW_PATH.exists():
        log("Raw dataset missing -> generating synthetic data")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_dataset.generate().to_csv(config.RAW_PATH, index=False)

    log("[1/6] Cleaning data")
    raw = data_cleaning.load_raw()
    df, cleaning_report = data_cleaning.clean_and_save()

    log("[2/6] Feature engineering + RFE + priority score")
    df = scoring.add_priority_score(features.build_features(df))

    log("[3/6] Training and evaluating models (takes ~15-30 s)")
    out = model.train_and_evaluate(df)
    metrics = out["results"]
    df["conversion_probability"] = out["oof_proba"].round(4)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SCORED_PATH, index=False)
    out["lift"].round(4).to_csv(config.REPORTS_DIR / "lift_table.csv", index=False)
    visuals.channel_summary(df).to_csv(config.REPORTS_DIR / "channel_summary.csv")
    out["importance"].round(5).to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    log("[4/6] Drawing charts")
    paths = [
        visuals.plot_missing_values(raw), visuals.plot_leads_by_source(df),
        visuals.plot_conversion_by_source(df), visuals.plot_budget_distribution(df),
        visuals.plot_city_property_heatmap(df), visuals.plot_engagement_vs_conversion(df),
        visuals.plot_response_time(df), visuals.plot_correlation(df), visuals.plot_rfe_segments(df),
        visuals.plot_priority_bands(df), visuals.plot_channel_overview(df),
        visuals.plot_roc(out["curves"], metrics), visuals.plot_confusion(metrics["confusion_matrix_rf"]),
        visuals.plot_feature_importance(out["importance"]), visuals.plot_lift(out["lift"]),
        visuals.plot_model_comparison(metrics),
    ]
    log("[5/6] Drawing diagrams")
    paths += visuals.make_all_diagrams()

    log("[6/6] Writing business recommendations")
    text = generate_report.build_report(df, metrics, out["importance"], out["lift"])
    (config.REPORTS_DIR / "BUSINESS_RECOMMENDATIONS.md").write_text(text, encoding="utf-8")

    log(f"Done. {len(paths)} images in {config.IMAGES_DIR}, outputs in {config.REPORTS_DIR}")
    return {"df": df, "metrics": metrics, "cleaning": cleaning_report}


if __name__ == "__main__":
    res = run_all()
    print(json.dumps({k: v for k, v in res["metrics"]["models"].items()}, indent=1))
