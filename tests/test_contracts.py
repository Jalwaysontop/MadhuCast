from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.contracts import Forecast, Meal, PatientProfile, PatientState, Provenance, Scenario

NOW = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)


def profile(**kw):
    base = dict(
        patient_id="P-017",
        cohort="t2d",
        age=52,
        sex="M",
        bmi=28.4,
        hba1c=8.1,
        fasting_glucose=162,
        provenance={"age": Provenance.MOCK},
    )
    return PatientProfile(**(base | kw))


def history(hours=4):
    t = pd.date_range(NOW - timedelta(hours=hours), NOW, freq="5min", tz="UTC")
    n = len(t)
    return pd.DataFrame(
        {
            "t": t,
            "glucose": np.full(n, 140.0),
            "hr": 72.0,
            "mets": 1.2,
            "sleep_proxy": 0.0,
            "missing_mask": False,
        }
    )


def forecast(**kw):
    base = dict(
        glucose_30=(130, 145, 160),
        glucose_60=(125, 150, 175),
        glucose_120=(110, 140, 170),
        spike_prob=0.1,
        hypo_prob=0.01,
        confidence=0.8,
    )
    return Forecast(**(base | kw))


def test_scenario_has_no_medication_field():
    fields = set(Scenario.model_fields)
    assert not {f for f in fields if "med" in f or "insulin" in f or "dose" in f}
    with pytest.raises(ValidationError):
        Scenario(medication_change="stop metformin")


def test_scenario_default_is_baseline_and_bounds():
    assert Scenario().is_baseline()
    assert not Scenario(walk_minutes=20).is_baseline()
    with pytest.raises(ValidationError):
        Scenario(portion_scale=3.0)
    with pytest.raises(ValidationError):
        Scenario(meal_time_shift_min=500)


def test_profile_rejects_real_names_and_unlabelled_grs():
    with pytest.raises(ValidationError):
        profile(patient_id="Ramesh Kumar")
    with pytest.raises(ValidationError):
        profile(genetic_risk_score=0.7)  # missing SYNTHETIC provenance
    p = profile(genetic_risk_score=0.7, provenance={"genetic_risk_score": "SYNTHETIC"})
    assert p.genetic_risk_score == 0.7


def test_state_validates_history():
    PatientState(profile=profile(), history=history(), meals=[], now=NOW)
    with pytest.raises(ValidationError):
        PatientState(profile=profile(), history=history(hours=2), meals=[], now=NOW)
    with pytest.raises(ValidationError):
        PatientState(profile=profile(), history=history().drop(columns="mets"), meals=[], now=NOW)
    with pytest.raises(ValidationError):
        PatientState(profile=profile(), history=history(), meals=[], now=NOW.replace(tzinfo=None))


def test_meal_requires_tz():
    m = dict(label="dinner", carbs_g=60, protein_g=10, fat_g=8, fibre_g=4, gl=30, provenance="MOCK")
    Meal(t=NOW, **m)
    with pytest.raises(ValidationError):
        Meal(t=NOW.replace(tzinfo=None), **m)


def test_forecast_quantile_order_and_ranges():
    assert forecast().horizon(60) == (125, 150, 175)
    with pytest.raises(ValidationError):
        forecast(glucose_30=(160, 145, 130))
    with pytest.raises(ValidationError):
        forecast(spike_prob=1.2)
    with pytest.raises(ValidationError):
        forecast(glucose_60=(5, 150, 175))
    with pytest.raises(ValidationError):
        forecast(unexpected_field=1)


def test_forecast_trajectory_validation():
    good = [
        {"t_min": 5, "q10": 130, "q50": 140, "q90": 150},
        {"t_min": 10, "q05": 120, "q10": 128, "q50": 141, "q90": 155, "q95": 160},
    ]
    assert len(forecast(trajectory=good).trajectory) == 2
    with pytest.raises(ValidationError):
        forecast(trajectory=[{"t_min": 5, "q10": 150, "q50": 140, "q90": 160}])
    with pytest.raises(ValidationError):
        forecast(trajectory=list(reversed(good)))
