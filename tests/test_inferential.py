"""Tests for the inferential-multiverse modeling logic."""

import numpy as np
import pandas as pd
import pytest

from abcd_ksads import inferential as inf


# ---- enough (inclusion gate) ------------------------------------------------


def test_enough_row_and_positive_gates():
    assert inf.enough(pd.Series([1] * 50)) is False            # < 100 rows
    assert inf.enough(pd.Series([1] * 5 + [0] * 195)) is False  # < 10 positives
    assert inf.enough(pd.Series([1] * 30 + [0] * 170)) is True


def test_enough_subgroup_mask_requires_ten_positives_in_group():
    y = pd.Series([1] * 30 + [0] * 170)
    assert inf.enough(y, mask=pd.Series([True] * 5 + [False] * 195)) is False
    assert inf.enough(y, mask=pd.Series([True] * 30 + [False] * 170)) is True


# ---- eta2 (variance decomposition) ------------------------------------------


def test_eta2_partitions_variance_by_axis():
    # all variance in logOR is explained by 'status', none by 'informant'
    sub = pd.DataFrame({
        "logor": [0.0, 0.0, 2.0, 2.0],
        "status": ["A", "A", "B", "B"],
        "informant": ["p", "y", "p", "y"],
    })
    out = inf.eta2(sub)
    assert out["eta2_status"] == pytest.approx(1.0)
    assert out["eta2_informant"] == pytest.approx(0.0)


def test_eta2_zero_total_variance_is_nan():
    sub = pd.DataFrame({
        "logor": [1.0, 1.0, 1.0],
        "status": ["A", "B", "A"],
        "informant": ["p", "y", "p"],
    })
    out = inf.eta2(sub)
    assert np.isnan(out["eta2_status"]) and np.isnan(out["eta2_informant"])


# ---- bucket_of --------------------------------------------------------------


def test_bucket_of_classifies_predictor_labels():
    assert inf.bucket_of("Female (vs male)") == "Sex"
    assert inf.bucket_of("Income (per SD)") == "Income"
    assert inf.bucket_of("Race: Asian vs White") == "Race/ethnicity"
    assert inf.bucket_of("DMN within-network FC (per SD)") == "Neuroimaging"
    assert inf.bucket_of("Screen time (per SD)") == "Culture/environment"


# ---- outcome_frame ----------------------------------------------------------


def test_outcome_frame_maps_binary_and_drops_not_administered():
    P = pd.DataFrame({"sex_f": [0.0, 1.0, 0.0]}, index=["p1", "p2", "p3"])
    stat = pd.Series({"p1": "positive", "p2": "administered_negative",
                      "p3": "not_administered"})
    df = inf.outcome_frame(P, stat)
    assert set(df.index) == {"p1", "p2"}       # not_administered dropped
    assert df.loc["p1", "y"] == 1
    assert df.loc["p2", "y"] == 0


def test_outcome_frame_none_when_status_empty():
    P = pd.DataFrame({"sex_f": [0.0]}, index=["p1"])
    assert inf.outcome_frame(P, None) is None
    assert inf.outcome_frame(P, pd.Series([], dtype=object)) is None


# ---- fit_adj (logistic GLM engine) ------------------------------------------


def _assoc_frame(n=500, beta=1.5, seed=0):
    """A modeling frame where outcome y is logistic in focal predictor x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-beta * x)))
    return pd.DataFrame({
        "y": y.astype(float),
        "x": x,
        "sex_f": rng.integers(0, 2, n).astype(float),
        "age_z": rng.normal(size=n),
        "site": rng.choice(["1", "2", "3"], n),
        "family_id": np.arange(n),
    })


def test_fit_adj_recovers_positive_association():
    orr, p = inf.fit_adj(_assoc_frame(beta=1.5), ["x"])["x"]
    assert orr > 1.0          # positive logistic slope -> OR > 1
    assert p < 0.05


def test_fit_adj_returns_nan_on_failure():
    # missing site/family_id columns -> exception inside the fit -> (nan, nan)
    d = pd.DataFrame({"y": [0.0, 1.0], "x": [0.0, 1.0]})
    orr, p = inf.fit_adj(d, ["x"])["x"]
    assert np.isnan(orr) and np.isnan(p)


def test_fit_adj_clamps_implausible_odds_ratio_to_nan():
    n = 200
    d = pd.DataFrame({
        "y": np.array([0] * 100 + [1] * 100, dtype=float),
        "x": np.array([0.0] * 100 + [1.0] * 100),  # perfect separation
        "sex_f": np.tile([0.0, 1.0], 100),
        "age_z": np.zeros(n),
        "site": ["1"] * n,
        "family_id": np.arange(n),
    })
    orr, p = inf.fit_adj(d, ["x"])["x"]
    assert np.isnan(orr)      # OR outside 0.02-50 (or fit fails) -> nan


def test_fit_adj_sex_focal_does_not_duplicate_column():
    d = _assoc_frame(seed=1)
    d = d.rename(columns={"x": "unused"})
    # make y depend on sex so the model is well behaved
    rng = np.random.default_rng(2)
    d["y"] = rng.binomial(1, np.where(d.sex_f == 1, 0.6, 0.3)).astype(float)
    res = inf.fit_adj(d, ["sex_f"])
    assert "sex_f" in res
    assert np.isfinite(res["sex_f"][0])   # no duplicate-column crash


# ---- fit_spec (all predictor buckets for one spec) --------------------------


def _full_frame(n=600, seed=1, race=None, y=None):
    rng = np.random.default_rng(seed)
    if race is None:
        race = rng.choice([inf.RACE_REF] + inf.RACE_LEVELS, n)
    df = pd.DataFrame({
        "y": (rng.binomial(1, 0.35, n).astype(float) if y is None else np.asarray(y, float)),
        "sex_f": rng.integers(0, 2, n).astype(float),
        "age_z": rng.normal(size=n),
        "income_z": rng.normal(size=n),
        "screentime_z": rng.normal(size=n),
        "fam_conflict_z": rng.normal(size=n),
        "scanner": rng.choice(["A", "B"], n),
        "mean_fd_z": rng.normal(size=n),
        "Race": race,
        "site": rng.choice(["1", "2", "3"], n),
        "family_id": np.arange(n),
    }, index=[f"p{i}" for i in range(n)])
    for fc, _ in inf.NEURAL:
        df[fc] = rng.normal(size=n)
    return df


def test_fit_spec_covers_all_predictor_buckets():
    out = inf.fit_spec(_full_frame())
    expected = {"Female (vs male)", "Income (per SD)",
                "Screen time (per SD)", "Family conflict (per SD)"}
    expected |= {lab for _, lab in inf.NEURAL}
    expected |= {f"Race: {lvl} vs {inf.RACE_REF}" for lvl in inf.RACE_LEVELS}
    assert expected <= set(out)
    assert all(isinstance(v, tuple) and len(v) == 2 for v in out.values())


def test_fit_spec_gates_out_small_samples():
    assert inf.fit_spec(_full_frame().head(50)) == {}   # < 100 rows -> nothing fit


def test_fit_spec_blanks_race_level_with_too_few_positives():
    n = 400
    race = np.array(["White"] * 150 + ["Black/AA"] * 100 + ["Hispanic"] * 100
                    + ["Asian"] * 30 + ["Other/Multiracial"] * 20)
    rng = np.random.default_rng(3)
    y = np.where(race == "Asian", 0.0,                       # Asian: zero positives
                 rng.binomial(1, 0.4, n).astype(float))
    out = inf.fit_spec(_full_frame(n=n, seed=3, race=race, y=y))
    assert np.isnan(out["Race: Asian vs White"][0])         # < 10 positives -> blanked
    # a well-populated level is still estimated
    assert np.isfinite(out["Race: Black/AA vs White"][0])


# ---- summarize_specs --------------------------------------------------------


def test_summarize_specs_flags_sign_flip_and_drops_all_nan_pairs():
    res = pd.DataFrame([
        dict(bucket="Sex", construct="dep", construct_label="Depression",
             predictor="Female (vs male)", status="current", informant="parent",
             OR=0.5, p=0.01, sig=True, logor=np.log(0.5)),
        dict(bucket="Sex", construct="dep", construct_label="Depression",
             predictor="Female (vs male)", status="ever_met", informant="youth",
             OR=2.0, p=0.20, sig=False, logor=np.log(2.0)),
        dict(bucket="Income", construct="dep", construct_label="Depression",
             predictor="Income (per SD)", status="current", informant="parent",
             OR=1.5, p=0.01, sig=True, logor=np.log(1.5)),
        dict(bucket="Income", construct="dep", construct_label="Depression",
             predictor="Income (per SD)", status="ever_met", informant="youth",
             OR=1.8, p=0.01, sig=True, logor=np.log(1.8)),
        dict(bucket="Sex", construct="adhd", construct_label="ADHD",
             predictor="Female (vs male)", status="current", informant="parent",
             OR=np.nan, p=np.nan, sig=False, logor=np.nan),
    ])
    S = inf.summarize_specs(res)
    byp = {(r.construct, r.predictor): r for _, r in S.iterrows()}
    assert ("adhd", "Female (vs male)") not in byp        # all-NaN pair dropped
    flip = byp[("dep", "Female (vs male)")]
    assert bool(flip.sign_flip) is True
    assert flip.OR_min == 0.5 and flip.OR_max == 2.0 and flip.n_specs == 2
    assert bool(byp[("dep", "Income (per SD)")].sign_flip) is False
    assert "eta2_status" in S.columns and "eta2_informant" in S.columns


# ---- build_specs (grid orchestration) ---------------------------------------


def test_build_specs_assembles_rows_with_sig_and_logor(monkeypatch):
    P = pd.DataFrame({"sex_f": [0.0, 1.0, 0.0]}, index=["p1", "p2", "p3"])
    stat = pd.Series({"p1": "positive", "p2": "administered_negative", "p3": "positive"})
    monkeypatch.setattr(inf, "construct_status", lambda *a, **k: stat)
    monkeypatch.setattr(inf, "fit_spec",
                        lambda df: {"Female (vs male)": (1.5, 0.01),
                                    "Income (per SD)": (np.nan, np.nan)})
    res = inf.build_specs(
        P, cache={}, constructs=[("depression", "Depression")],
        informants=["parent"], statuses=["current"], thresholds=[False],
    )
    assert set(res.predictor) == {"Female (vs male)", "Income (per SD)"}
    fem = res[res.predictor == "Female (vs male)"].iloc[0]
    assert fem.bucket == "Sex" and fem.OR == 1.5 and bool(fem.sig) is True
    assert fem.logor == pytest.approx(np.log(1.5))
