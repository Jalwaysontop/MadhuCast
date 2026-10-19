"""Property tests on the mock predictor: directional sanity the real model must also pass."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.contracts import Meal, PatientProfile, PatientState, Provenance, Scenario
from src.serving import backend, mock_data, mock_predict
from src.serving import timeseries as ts

NOW = datetime(2026, 11, 19, 14, 30, tzinfo=UTC)


def make_state(gl=40.0, fibre=3.0, fat=8.0, hba1c=7.5, bmi=28.0, sleep_h=7.0, glucose=140.0, cohort="t2d"):
    t = pd.date_range(NOW - timedelta(hours=16), NOW, freq="5min", tz="UTC")
    local_hour = t.tz_convert(ts.IST).hour + t.tz_convert(ts.IST).minute / 60
    wake = 6.5
    asleep = (
        (local_hour < wake) & (local_hour >= wake - sleep_h)
        if sleep_h <= wake
        else ((local_hour < wake) | (local_hour >= 24 - (sleep_h - wake)))
    )
    hist = pd.DataFrame(
        {
            "t": t,
            "glucose": glucose,
            "hr": 65.0,
            "mets": np.where(asleep, 0.95, 1.3),
            "sleep_proxy": asleep.astype(float),
            "missing_mask": False,
        }
    )
    prof = PatientProfile(
        patient_id="P-900",
        cohort=cohort,
        age=50,
        sex="F",
        bmi=bmi,
        hba1c=hba1c,
        fasting_glucose=glucose,
        provenance={"age": Provenance.MOCK},
    )
    meal = Meal(
        t=NOW,
        label="Lunch",
        carbs_g=gl * 1.4,
        protein_g=10,
        fat_g=fat,
        fibre_g=fibre,
        gl=gl,
        provenance=Provenance.MOCK,
    )
    return PatientState(profile=prof, history=hist, meals=[meal], now=NOW, data_quality=1.0)


def peak(state, scenario=None):
    return max(p.q50 for p in backend.predict(state, scenario, explain=False).trajectory)


profiles = dict(
    hba1c=st.floats(5.0, 11.0),
    bmi=st.floats(19, 38),
    glucose=st.floats(85, 190),
    gl=st.floats(5, 90),
    fibre=st.floats(0, 12),
    fat=st.floats(0, 30),
)


@settings(max_examples=40, deadline=None)
@given(**profiles, extra=st.floats(5, 60))
def test_more_carbs_raises_peak(hba1c, bmi, glucose, gl, fibre, fat, extra):
    lo = make_state(gl=gl, fibre=fibre, fat=fat, hba1c=hba1c, bmi=bmi, glucose=glucose)
    hi = make_state(gl=gl + extra, fibre=fibre, fat=fat, hba1c=hba1c, bmi=bmi, glucose=glucose)
    assert peak(hi) > peak(lo)


@settings(max_examples=40, deadline=None)
@given(**profiles, minutes=st.integers(5, 60))
def test_walk_lowers_or_keeps_peak(hba1c, bmi, glucose, gl, fibre, fat, minutes):
    s = make_state(gl=gl, fibre=fibre, fat=fat, hba1c=hba1c, bmi=bmi, glucose=glucose)
    assert peak(s, Scenario(walk_minutes=minutes)) <= peak(s) + 1e-6


@settings(max_examples=40, deadline=None)
@given(**profiles, short=st.floats(3, 6.5))
def test_shorter_sleep_raises_or_keeps_peak(hba1c, bmi, glucose, gl, fibre, fat, short):
    s = make_state(gl=gl, fibre=fibre, fat=fat, hba1c=hba1c, bmi=bmi, glucose=glucose)
    assert peak(s, Scenario(prior_sleep_hours=short)) >= peak(s, Scenario(prior_sleep_hours=8.0)) - 1e-6


@settings(max_examples=40, deadline=None)
@given(**profiles)
def test_quantiles_ordered_and_bounded(hba1c, bmi, glucose, gl, fibre, fat):
    fc = backend.predict(make_state(gl=gl, fibre=fibre, fat=fat, hba1c=hba1c, bmi=bmi, glucose=glucose))
    for p in fc.trajectory:
        assert p.q05 <= p.q10 <= p.q50 <= p.q90 <= p.q95
        assert p.q05 >= 40 and p.q95 <= 400
    for h in (30, 60, 120):
        q = fc.horizon(h)
        assert q[0] <= q[1] <= q[2]


def test_millet_swap_lowers_peak_for_every_rice_meal():
    df, _ = backend.load_report("whatif_sanity")
    row = df.set_index("test").loc["millet_swap_lowers_peak"]
    assert row["n_cases"] >= 3 and row["pass_rate"] == 1.0
    assert (df["pass_rate"] == 1.0).all()


def test_cohort_shape_and_special_patients():
    pids = mock_data.list_patients()
    assert len(pids) == 12
    cohorts = [mock_data.get_state(p).profile.cohort for p in pids]
    assert cohorts.count("healthy") == cohorts.count("prediabetes") == cohorts.count("t2d") == 4
    p7 = mock_data.get_state("P-007", mock_data.DEMO_NOW)
    assert ts.glucose_now(p7.history, p7.now) is None and ts.stale_minutes(p7.history, p7.now) >= 40
    p11 = mock_data.get_state("P-011", mock_data.DEMO_NOW)
    fc = backend.predict(p11)
    assert fc.glucose_30[0] < 70 and fc.hypo_prob > 0.3


def test_mock_is_deterministic():
    a = backend.predict(mock_data.get_state("P-010"))
    b = backend.predict(mock_data.get_state("P-010"))
    assert a == b


def test_attributions_are_faithful_for_meal():
    s = make_state(gl=60)
    fc = backend.predict(s)
    no_meal = s.model_copy(update={"meals": []})
    assert abs(fc.shap_values["meal_gl"] - (peak(s) - peak(no_meal))) < 1.0
    assert fc.provenance == Provenance.MOCK and fc.model_version == mock_predict.MODEL_VERSION


def test_resolve_meal_units_and_errors():
    m = mock_data.resolve_meal(
        [{"food": "wheat roti", "portion": 2, "unit": "piece"}, {"food": "dal (toor)", "portion": 150, "unit": "g"}]
    )
    assert m.carbs_g == 2 * 18 + 18 and m.provenance == Provenance.MOCK
    rice = mock_data.resolve_meal([{"food": "white rice", "portion": 1, "unit": "katori"}])
    assert rice.assumption_flags
    for bad in ([{"food": "pizza", "portion": 1}], [{"food": "idli", "portion": 1, "unit": "katori"}], []):
        try:
            mock_data.resolve_meal(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
