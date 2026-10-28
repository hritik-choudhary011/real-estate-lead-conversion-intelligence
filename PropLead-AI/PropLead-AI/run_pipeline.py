"""Run the whole PropLead AI pipeline end to end.

    python run_pipeline.py            # generate data if missing, clean, score, model, charts, report
    python run_pipeline.py --regen    # regenerate the synthetic dataset first
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config, data_cleaning, features, generate_dataset, model, scoring, visuals
from src.generate_report import save_report


def run_all(regen: bool = False, make_charts: bool = True, verbose: bool = True) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)

    if regen or not config.RAW_PATH.exists():
        log("[1/7] Generating synthetic dataset ...")
        config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset.generate().to_csv(config.RAW_PATH, index=False)
    else:
        log("[1/7] Using existing dataset:", config.RAW_PATH.name)

    raw = data_cleaning.load_raw()
    log("[2/7] Cleaning ...")
    df, report = data_cleaning.clean_and_save()
    log(f"      rows {report['rows_in']} -> {report['rows_out']}, category fixes {report['category_fixes']}")

    log("[3/7] Feature engineering + RFE + priority score ...")
    df = scoring.add_priority_score(features.build_features(df))

    log("[4/7] Training and evaluating models (takes ~15 s) ...")
    out = model.train_and_evaluate(df)
    df["conversion_probability"] = out["oof_proba"].round(4)
    df["probability_decile"] = pd.qcut(df["conversion_probability"].rank(method="first", ascending=False),
                                       10, labels=range(1, 11)).astype(int)
    df.to_csv(config.SCORED_PATH, index=False)

    if make_charts:
        log("[5/7] Drawing charts ...")
        visuals.plot_missing_values(raw)
        visuals.plot_leads_by_source(df)
        visuals.plot_conversion_by_source(df)
        visuals.plot_budget_distribution(df)
        visuals.plot_city_property_heatmap(df)
        visuals.plot_engagement_vs_conversion(df)
        visuals.plot_response_time(df)
        visuals.plot_correlation(df)
        visuals.plot_rfe_segments(df)
        visuals.plot_priority_bands(df)
        visuals.plot_channel_overview(df)
        visuals.plot_roc(out["curves"], out["results"])
        visuals.plot_confusion(out["results"]["confusion_matrix_rf"])
        visuals.plot_feature_importance(out["importance"])
        visuals.plot_lift(out["lift"])
        visuals.plot_model_comparison(out["results"])
        log("[6/7] Drawing diagrams ...")
        visuals.make_all_diagrams()

    log("[7/7] Writing business recommendations ...")
    save_report(df, out["results"], out["lift"], out["importance"])
    out["importance"].to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    out["lift"].to_csv(config.REPORTS_DIR / "lift_table.csv", index=False)
    visuals.channel_summary(df).to_csv(config.REPORTS_DIR / "channel_summary.csv")
    log("Done. See data/processed, images, reports, models.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate the synthetic dataset first")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args()
    run_all(regen=a.regen, make_charts=not a.no_charts)
