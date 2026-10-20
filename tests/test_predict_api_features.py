from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.serving.predict_api import build_feature_matrix


def test_patient_state_maps_to_514_features_in_metadata_order():
    now = pd.Timestamp("2026-10-20T12:00:00Z")
    times = pd.date_range(end=now, periods=73, freq="5min", tz="UTC")
    history = pd.DataFrame({
        "t": times,
        "glucose": np.linspace(115.0, 130.0, len(times)),
        "hr": np.full(len(times), 72.0),
        "mets": np.full(len(times), 1.3),
        "sleep_proxy": np.zeros(len(times)),
        "missing_mask": np.zeros(len(times), dtype=bool),
    })
    profile = SimpleNamespace(
        sex="F", age=55, bmi=29.0, hba1c=7.2, fasting_glucose=132.0,
        lipids={"tg": 160.0, "chol": 195.0, "hdl": 45.0, "ldl": 115.0},
    )
    state = SimpleNamespace(profile=profile, history=history, meals=[], now=now.to_pydatetime())
    metadata = {
        "n_features": 514,
        "sequence_columns": ["glucose", "hr", "mets", "carbs", "protein", "fat", "fiber"],
        "static_columns": [
            "sex_male", "age", "bmi", "hba1c", "fasting_glu", "fasting_insulin",
            "tg", "chol", "hdl", "ldl",
        ],
    }

    features = build_feature_matrix(state, metadata)

    assert features.shape == (1, 514)
    assert np.isfinite(features[0, :504]).all()
    assert features[0, 504] == 0.0       # sex_male
    assert features[0, 505] == 55.0      # age
    assert np.isnan(features[0, 509])    # fasting_insulin unavailable in contract
    assert features[0, 510] == 160.0     # triglycerides
    assert features[0, 511] == 195.0     # total cholesterol


def test_future_meals_are_not_added_to_history_features():
    now = pd.Timestamp("2026-10-20T12:00:00Z")
    times = pd.date_range(end=now, periods=73, freq="5min", tz="UTC")
    history = pd.DataFrame({
        "t": times, "glucose": 120.0, "hr": 70.0, "mets": 1.2,
        "sleep_proxy": 0.0, "missing_mask": False,
    })
    profile = SimpleNamespace(
        sex="M", age=44, bmi=25.0, hba1c=6.8, fasting_glucose=118.0, lipids=None,
    )
    future_meal = SimpleNamespace(
        t=(now + pd.Timedelta(minutes=30)).to_pydatetime(),
        carbs_g=80.0, protein_g=20.0, fat_g=10.0, fibre_g=5.0,
    )
    state = SimpleNamespace(profile=profile, history=history, meals=[future_meal], now=now.to_pydatetime())
    metadata = {
        "n_features": 514,
        "sequence_columns": ["glucose", "hr", "mets", "carbs", "protein", "fat", "fiber"],
        "static_columns": [
            "sex_male", "age", "bmi", "hba1c", "fasting_glu", "fasting_insulin",
            "tg", "chol", "hdl", "ldl",
        ],
    }

    features = build_feature_matrix(state, metadata)
    sequence = features[0, :504].reshape(72, 7)
    assert np.all(sequence[:, 3:] == 0.0)
