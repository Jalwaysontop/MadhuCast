from datetime import timedelta

import pandas as pd
import pytest

from src.contracts import Scenario
from src.serving import backend, mock_data, scenario
from src.serving import timeseries as ts


@pytest.fixture
def dinner():
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    return st, len(st.meals) - 1


def apply(st, sc):
    return scenario.apply(st, sc, backend.resolve_meal)


def frames_equal(a, b, cols):
    return a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))


def test_baseline_is_identity(dinner):
    st, _ = dinner
    assert apply(st, Scenario()) is st


def test_portion_changes_only_target_meal(dinner):
    st, i = dinner
    out = apply(st, Scenario(portion_scale=0.5))
    assert out.meals[i].gl == pytest.approx(st.meals[i].gl * 0.5)
    assert out.meals[:i] == st.meals[:i]
    assert out.history.equals(st.history) and out.planned_activity == []


def test_swap_rewrites_items_keeps_time(dinner):
    st, i = dinner
    items = scenario.rice_to_millet(st.meals[i].items)
    out = apply(st, Scenario(meal_swap={"meal_index": i, "items": items}))
    assert out.meals[i].t == st.meals[i].t
    assert all("rice" not in it["food"] for it in out.meals[i].items)
    assert out.meals[i].gl < st.meals[i].gl
    assert "(swap)" in out.meals[i].label
    assert out.history.equals(st.history)


def test_walk_changes_only_activity(dinner):
    st, i = dinner
    out = apply(st, Scenario(walk_minutes=20))
    assert out.meals == st.meals
    assert len(out.planned_activity) == 1
    blk = out.planned_activity[0]
    assert blk.start == st.meals[i].t + timedelta(minutes=15) and blk.minutes == 20
    assert frames_equal(out.history, st.history, ["t", "glucose", "sleep_proxy", "missing_mask"])


def test_sleep_changes_only_night_window(dinner):
    st, _ = dinner
    out = apply(st, Scenario(prior_sleep_hours=8.0))
    assert ts.prior_sleep_hours(out.history, out.now) == pytest.approx(8.0)
    start, end = ts.night_window(st.now)
    t = pd.to_datetime(st.history["t"], utc=True)
    outside = (t < pd.Timestamp(start)) | (t >= pd.Timestamp(end))
    assert frames_equal(out.history[outside], st.history[outside], list(st.history.columns))
    assert frames_equal(out.history, st.history, ["t", "glucose", "missing_mask"])
    assert out.meals == st.meals


def test_later_meal_becomes_planned(dinner):
    st, i = dinner
    out = apply(st, Scenario(meal_time_shift_min=60))
    assert out.meals[i].t == st.meals[i].t + timedelta(minutes=60) > out.now
    assert out.now == st.now


def test_earlier_meal_moves_evaluation_time(dinner):
    st, i = dinner
    out = apply(st, Scenario(meal_time_shift_min=-60))
    assert out.now == st.meals[i].t - timedelta(minutes=60)
    assert pd.to_datetime(out.history["t"], utc=True).max() <= pd.Timestamp(out.now)
    assert all(m.t <= out.now for m in out.meals)


def test_presets_and_compare(dinner):
    st, i = dinner
    base = backend.predict(st)
    for name in scenario.PRESETS:
        sc = scenario.preset(name, st, i)
        _, fc = backend.run_scenario(st, sc)
        c = scenario.compare(base, fc)
        if name == "Short sleep (5 h)":
            assert c.d_peak >= 0
        else:
            assert c.d_peak < 0
    c = scenario.compare(base, backend.predict(st, scenario.preset("Combined", st, i)))
    assert not c.bands_overlap and c.overlap_note is None
    c2 = scenario.compare(base, backend.predict(st, Scenario(walk_minutes=5)))
    assert c2.bands_overlap and "may not be meaningful" in c2.overlap_note


def test_preset_not_applicable():
    st = mock_data.get_state("P-001", mock_data.DEMO_NOW)
    with pytest.raises(ValueError, match="no rice"):
        scenario.preset("Rice → millet roti", st, len(st.meals) - 1)
    with pytest.raises(ValueError):
        scenario.preset("Nonsense", st, 0)


def test_whatif_recompute_is_fast(dinner):
    import time

    st, i = dinner
    sc = scenario.preset("Combined", st, i)
    t0 = time.perf_counter()
    for _ in range(5):
        backend.run_scenario(st, sc)
    assert (time.perf_counter() - t0) / 5 < 0.3
