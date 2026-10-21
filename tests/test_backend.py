"""Backend adapter tests, plus the contract suite that gates the Day-10/Day-14 handoffs.

`pytest -m contract` runs TestContract against MADHUTWIN_BACKEND=real.
"""

import pandas as pd
import pytest

from src.contracts import FOOD_DB_COLUMNS, REPORT_FILES, ContractError, Forecast, Meal, PatientState
from src.serving import backend


def test_mode_selection(monkeypatch):
    assert backend.describe() == {"data": "MOCK", "model": "MOCK"}
    monkeypatch.setenv("MADHUTWIN_DATA", "real")
    assert backend.describe() == {"data": "REAL", "model": "MOCK"}
    monkeypatch.setenv("MADHUTWIN_MODEL", "nonsense")
    with pytest.raises(ValueError):
        backend.model_mode()


def test_real_data_missing_raises_contract_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MADHUTWIN_DATA", "real")
    monkeypatch.setattr(backend, "PROFILES_DIR", tmp_path / "nope")
    with pytest.raises(ContractError, match="P3:patient_profiles"):
        backend.list_patients()


def test_real_report_missing_is_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("MADHUTWIN_MODEL", "real")
    monkeypatch.setattr(backend, "REPORTS_DIR", tmp_path)
    assert backend.load_report("ablation") == (None, "MISSING")
    thr, calibrated = backend.thresholds()
    assert not calibrated and "strata" in thr


def test_real_report_bad_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("MADHUTWIN_MODEL", "real")
    monkeypatch.setattr(backend, "REPORTS_DIR", tmp_path)
    pd.DataFrame({"variant": ["x"]}).to_csv(tmp_path / "ablation.csv", index=False)
    with pytest.raises(ContractError, match="missing columns"):
        backend.load_report("ablation")


def test_real_predict_validates_payload(monkeypatch):
    class Bad:
        @staticmethod
        def predict(state, scenario):
            return {"glucose_30": (200, 150, 100)}

    monkeypatch.setenv("MADHUTWIN_DATA", "mock")
    monkeypatch.setenv("MADHUTWIN_MODEL", "real")
    monkeypatch.setattr(backend, "_real_predictor", lambda: Bad)
    st = backend.get_state(backend.list_patients()[0])
    with pytest.raises(ContractError, match="P2:predict"):
        backend.predict(st)


def test_mock_reports_all_load():
    for name in REPORT_FILES:
        payload, label = backend.load_report(name)
        assert payload is not None and label == "MOCK"


class TestContract:
    """Same assertions for mock (default) and real (`pytest -m contract`)."""

    def _check_all(self):
        pids = backend.list_patients()
        assert pids
        st = backend.get_state(pids[0])
        assert isinstance(st, PatientState)
        fc = backend.predict(st)
        assert isinstance(fc, Forecast)
        db = backend.food_db()
        assert set(FOOD_DB_COLUMNS) <= set(db.columns)
        first = db.iloc[0]
        meal = backend.resolve_meal([{"food": first["food"], "portion": 1, "unit": first["portion_unit"]}])
        assert isinstance(meal, Meal) and meal.carbs_g >= 0
        for name in REPORT_FILES:
            backend.load_report(name)

    def test_contract_mock(self):
        self._check_all()

    @pytest.mark.contract
    def test_contract_real(self):
        backend.clear_caches()
        self._check_all()
