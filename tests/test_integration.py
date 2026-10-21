"""P2/P3 integration: the meal-engine bridge, the contract-file export, and the hybrid real model."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from src.contracts import FOOD_DB_COLUMNS, ContractError, Meal, PatientProfile, Provenance, Scenario
from src.serving import backend, mock_data, p3_bridge, scenario

# --------------------------------------------------------------------------- P3 meal engine bridge


def test_bridge_food_db_matches_contract():
    db = p3_bridge.food_db()
    assert list(db.columns[: len(FOOD_DB_COLUMNS)]) == list(FOOD_DB_COLUMNS)
    assert db["food"].is_unique and (db["food"] == db["food"].str.lower()).all()
    assert (db[["carbs_g", "protein_g", "fat_g", "fibre_g", "gl_per_unit"]] >= 0).all().all()
    # P3's preparation GI multipliers are assumptions, so they must surface as flags.
    rice = db.set_index("food").loc["white rice (steamed)"]
    assert "x0.95" in rice["assumption_flag"]


def test_bridge_resolve_uses_p3_nutrition():
    from src.data.meal_engine import IndianFoodDB, compute_meal_vector

    m = p3_bridge.resolve_meal([{"food": "Bajra Roti", "portion": 2, "unit": "piece"}, {"food": "D001", "portion": 1}])
    ref = compute_meal_vector([("Bajra Roti", 2), ("D001", 1)], IndianFoodDB(p3_bridge.FOOD_DB_PATH))
    assert isinstance(m, Meal) and m.provenance == Provenance.SYNTHETIC
    assert m.carbs_g == pytest.approx(ref["total_carbs_g"]) and m.gl == pytest.approx(ref["total_glycemic_load"])
    assert [i["food"] for i in m.items] == ["bajra roti", "toor dal (arhar)"]
    grams = p3_bridge.resolve_meal([{"food": "white rice (steamed)", "portion": 150, "unit": "g"}])
    assert grams.carbs_g == pytest.approx(44.2)


@pytest.mark.parametrize(
    "items,match",
    [
        ([{"food": "rice", "portion": 1}], "unknown food"),  # no fuzzy first-match
        ([{"food": "bajra roti", "portion": 1, "unit": "katori"}], "not valid"),
        ([{"food": "bajra roti", "portion": 0}], "positive"),
        ([], "at least one"),
    ],
)
def test_bridge_rejects_bad_items(items, match):
    with pytest.raises(ValueError, match=match):
        p3_bridge.resolve_meal(items)


def test_millet_preset_on_p3_names():
    items = [
        {"food": "white rice (steamed)", "portion": 1, "unit": "katori"},
        {"food": "toor dal (arhar)", "portion": 1},
    ]
    out = scenario.rice_to_millet(items, "bajra roti")
    assert out[0] == {"food": "bajra roti", "portion": 2, "unit": "piece"}
    assert p3_bridge.resolve_meal(out).carbs_g < p3_bridge.resolve_meal(items).carbs_g + 1


# --------------------------------------------------------------------------- contract-file export


def test_export_is_contract_valid(real_data):
    pids = backend.list_patients()
    assert pids == [e.pid for e in real_data] and all(p.startswith("P-") for p in pids)
    for pid in pids:
        st = backend.get_state(pid)
        assert st.data_quality is not None
        assert {m.provenance for m in st.meals} <= {Provenance.REAL, Provenance.SYNTHETIC}
    start, end, now = backend.time_window()
    assert start < end and now == end


def test_export_provenance_and_pairing(p3_exports):
    exports, real = p3_exports
    for e in exports:
        p = e.profile
        assert isinstance(p, PatientProfile)
        if p.pairing:  # T2D wearers carry P3's synthetic EHR
            assert p.cohort == "t2d" and p.provenance["hba1c"] == Provenance.SYNTHETIC
            assert p.provenance["genetic_risk_score"] == Provenance.SYNTHETIC
        if not real:
            assert set(p.provenance.values()) == {Provenance.SYNTHETIC}
    assert any(e.profile.pairing for e in exports) and any(not e.profile.pairing for e in exports)


def test_export_history_is_not_interpolated(p3_exports):
    from src.serving.p3_export import history, load_wearables

    data, _ = load_wearables()
    wid = sorted(data["cgm"]["participant_id"].unique())[0]
    drop = data["cgm"].index[data["cgm"]["participant_id"] == wid][100:110]
    gap = {**data, "cgm": data["cgm"].drop(drop)}
    h = history(gap, wid)
    assert int(h["missing_mask"].sum()) == 10 and h.loc[h["missing_mask"], "glucose"].isna().all()
    assert pd.Series(h["t"]).diff().dropna().eq(pd.Timedelta(minutes=5)).all()
    assert h["mets"].between(1, 12).all() and set(h["sleep_proxy"].unique()) <= {0.0, 1.0}


def test_export_meals_flag_assumptions(p3_exports):
    meals = [m for e in p3_exports[0] for m in e.meals]
    indian = [m for m in meals if m.items]
    assert indian and all(m.provenance == Provenance.SYNTHETIC for m in indian)
    assert all(any("substituted" in f for f in m.assumption_flags) for m in indian)
    assert all(any("GI 55" in f for f in m.assumption_flags) for m in meals if not m.items)
    # Itemised meals must resolve through the same bridge the what-if lab uses.
    for m in indian[:5]:
        assert p3_bridge.resolve_meal(m.items).carbs_g == pytest.approx(m.carbs_g, abs=0.5)


def test_export_write_replaces_old_files(p3_exports, tmp_path):
    from src.serving import p3_export

    stale = tmp_path / "patient_profiles" / "P-999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    p3_export.write(p3_exports[0], tmp_path)
    assert not stale.exists()
    meals = json.loads((tmp_path / "meals" / "P-001.json").read_text())
    assert all(Meal.model_validate(m) for m in meals)


# --------------------------------------------------------------------------- hybrid real model (P2)


@pytest.fixture
def real_model(monkeypatch):
    monkeypatch.setenv("MADHUTWIN_MODEL", "real")
    backend.clear_caches()
    yield
    backend.clear_caches()


def test_real_model_parts_are_disclosed(real_model):
    parts = backend.model_parts()
    assert parts["spike probability"] == "MODEL_ESTIMATE" and "MOCK" in parts.values()
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    fc = backend.predict(st)
    assert fc.provenance == Provenance.MOCK and fc.model_version.startswith("xgb-")
    assert backend.is_real_spike(fc) and not fc.shap_values


def test_real_model_short_history_falls_back(real_model):
    # P-007's CGM gap leaves the six-hour XGBoost input unresolved: one patient must not break the queue.
    st = mock_data.get_state("P-007", mock_data.DEMO_NOW)
    with pytest.raises(ContractError, match="P2:input"):
        backend._real_predictor().predict(st, Scenario())
    fc = backend.predict(st)
    assert fc.provenance == Provenance.MOCK and "real model skipped" in fc.model_version
    assert fc.detector_flags["classifier"] is False and not backend.is_real_spike(fc)


def test_real_model_other_contract_errors_still_raise(real_model, monkeypatch):
    class Broken:
        @staticmethod
        def predict(state, scenario):
            raise ContractError("P2:model", "artifact missing")

    monkeypatch.setattr(backend, "_real_predictor", lru_cache(maxsize=1)(lambda: Broken))
    with pytest.raises(ContractError, match="artifact missing"):
        backend.predict(mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW))


def test_spike_blind_levers(real_model):
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    fc = backend.predict(st)
    assert backend.spike_blind_levers(Scenario(portion_scale=0.5), fc) == []
    assert backend.spike_blind_levers(Scenario(walk_minutes=20, prior_sleep_hours=5), fc) == [
        "a post-meal walk",
        "last night's sleep",
    ]
    assert backend.spike_blind_levers(Scenario(meal_time_shift_min=60), fc) == ["a later meal time"]
    assert backend.spike_blind_levers(Scenario(meal_time_shift_min=-60), fc) == []
    # The classifier really is blind to a walk: its probability does not move.
    walk = backend.predict(st, Scenario(walk_minutes=30))
    assert walk.spike_prob == fc.spike_prob


def test_spike_blind_levers_off_for_mock():
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    assert backend.spike_blind_levers(Scenario(walk_minutes=20), backend.predict(st)) == []
    assert backend.model_parts() == {"forecast": "MOCK"}


def test_full_real_backend_predicts_every_patient(real_data, real_model):
    probs = []
    for pid in backend.list_patients():
        fc = backend.predict(backend.get_state(pid))
        probs.append(fc.spike_prob)
        assert 0 <= fc.spike_prob <= 1
    assert np.isfinite(probs).all()
    meal = backend.resolve_meal([{"food": "bajra roti", "portion": 2, "unit": "piece"}])
    assert meal.provenance == Provenance.SYNTHETIC


def test_data_badge_follows_provenance_not_mode(real_data):
    # P3's demo cohort under MADHUTWIN_DATA=real is SYNTHETIC, never badged REAL.
    assert backend.data_provenance() == "SYNTHETIC"


def test_data_badge_mock():
    assert backend.data_provenance() == "MOCK"
