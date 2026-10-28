"""Basic sanity tests. Run with:  pytest -q   (pip install pytest)"""
import pandas as pd

from src import config, data_cleaning, features, generate_dataset, model, scoring


def _raw(n=300, seed=1):
    return generate_dataset.generate(n=n, seed=seed)


def test_generator_shape_and_determinism():
    a, b = _raw(), _raw()
    assert a.shape == (300, 18)
    pd.testing.assert_frame_equal(a, b)
    assert a["lead_id"].is_unique


def test_no_personal_identifier_columns():
    banned = {"name", "phone", "email", "address", "customer_name", "mobile"}
    assert banned.isdisjoint(c.lower() for c in _raw().columns)


def test_cleaning_fixes_issues():
    raw = _raw()
    assert raw.isna().sum().sum() > 0
    df, rep = data_cleaning.clean(raw)
    assert df.isna().sum().sum() == 0
    assert set(df["lead_source"]) <= set(generate_dataset.SOURCE_SHARE)
    assert rep["rows_out"] == len(df) <= rep["rows_in"]


def test_cleaning_removes_duplicates():
    raw = _raw(100)
    dup = pd.concat([raw, raw.head(5)], ignore_index=True)
    df, rep = data_cleaning.clean(dup)
    assert rep["exact_duplicates_removed"] == 5 and len(df) == 100


def test_features_and_rfe_ranges():
    df = features.build_features(data_cleaning.clean(_raw())[0])
    for c in ["R_score", "F_score", "E_score"]:
        assert df[c].between(1, 5).all()
    assert df["RFE_total"].between(3, 15).all()
    assert df["response_band"].notna().all()


def test_priority_score_properties():
    assert sum(config.SCORE_WEIGHTS.values()) == 100
    df = scoring.add_priority_score(features.build_features(data_cleaning.clean(_raw())[0]))
    assert df["priority_score"].between(0, 100).all()
    # higher score never gets a lower band
    ordered = df.sort_values("priority_score")["priority_band"].cat.codes
    assert ordered.is_monotonic_increasing
    # components add up to the score (rounding tolerance)
    comp = scoring.score_components(df).sum(axis=1)
    assert (comp - df["priority_score"]).abs().max() < 0.1


def test_model_has_no_leakage_columns():
    for bad in config.EXCLUDED_FROM_MODEL:
        assert bad not in model.FEATURES
    assert "lead_status" not in model.FEATURES and "converted" not in model.FEATURES


def test_model_beats_baseline():
    df = scoring.add_priority_score(features.build_features(data_cleaning.clean(_raw(800, seed=3))[0]))
    out = model.train_and_evaluate(df, save=False)
    m = out["results"]["models"]
    assert m["Random Forest"]["roc_auc"] > m["Baseline (majority class)"]["roc_auc"] + 0.05
