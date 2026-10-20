"""Smoke-test the supplied native XGBoost JSON artifact. Run from project root."""
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_all_cgmacros.json"


def test_json_model_loads_and_predicts_probability():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    assert model.n_features_in_ == 514
    probabilities = model.predict_proba(np.zeros((1, 514), dtype=np.float32))
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert abs(float(probabilities.sum(axis=1)[0]) - 1.0) < 1e-5
