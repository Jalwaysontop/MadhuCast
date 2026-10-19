from src.contracts import Forecast
from src.explain import templates as T
from src.serving import backend, mock_data


def fc(**kw):
    base = dict(glucose_30=(150, 170, 190), glucose_60=(160, 190, 220), glucose_120=(140, 170, 200),
                spike_prob=0.643, hypo_prob=0.004, confidence=0.72)
    return Forecast(**(base | kw))


def test_rounding_and_units():
    assert T.round_mgdl(43.2) == 45 and T.round_mgdl(7.6) == 8 and T.round_mgdl(-12.4) == 10
    assert T.fmt_value("walk_minutes", 1) == "1 minute"
    assert T.fmt_value("walk_minutes", 20) == "20 minutes"
    assert T.fmt_value("sleep_hours_prev", 5.25) == "5.2 h"
    assert T.fmt_value("hba1c", 8.44) == "8.4%"
    assert T.fmt_value("glucose_now", 163.9) == "164 mg/dL"
    assert T.fmt_pct(0.004) == "<1%" and T.fmt_pct(0.999) == ">99%" and T.fmt_pct(0.643) == "64%"


def test_risk_sentence_snapshot():
    assert T.risk_sentence(fc()) == (
        "Estimated chance of glucose above 180 mg/dL within 2 hours: 64%; below 70 mg/dL: <1% "
        "(model estimate, confidence 72%).")


def test_contributor_sentences_snapshot():
    f = fc(shap_values={"meal_gl": 63.0, "sleep_hours_prev": 11.8, "walk_minutes": -0.4, "hba1c": 22.1},
           feature_values={"meal_gl": 68.4, "sleep_hours_prev": 5.2, "walk_minutes": 0, "hba1c": 8.4},
           feature_baselines={"sleep_hours_prev": 7.1})
    assert T.explain(f) == [
        "Recent meal glycaemic load (68) is raising the predicted peak by about 65 mg/dL.",
        "HbA1c (8.4%) is raising the predicted peak by about 20 mg/dL.",
        "Sleep last night (proxy) was 5.2 h vs the usual 7.1 h, raising the predicted peak by about 10 mg/dL.",
    ]


def test_lowering_and_little_effect():
    assert T.contributor_sentence("walk_minutes", -18.0, 20) == (
        "Walking (last 3 h and planned) (20 minutes) is lowering the predicted peak by about 20 mg/dL.")
    assert T.contributor_sentence("walk_minutes", 0.3, 0) == (
        "Walking (last 3 h and planned) (0 minutes) has little effect on the predicted peak.")


def test_unknown_feature_uses_raw_key():
    assert T.contributor_sentence("cgm_entropy", 12.0) == "cgm_entropy is raising the predicted peak by about 10 mg/dL."
    assert T.spec("cgm_entropy").group == "Other"


def test_fallbacks():
    assert T.explain(fc()) == [T.FALLBACK]
    assert T.explain(fc(shap_values={"meal_gl": 0.2})) == [
        "No single feature moves the predicted peak by more than 1 mg/dL."]


def test_counterfactual_sentence():
    s = T.counterfactual_sentence("the millet-roti alternative", (250, 290), (195, 230), d_peak=-52.4)
    assert s == ("Compared with the millet-roti alternative, this plan is predicted to raise peak glucose by "
                 "~50 mg/dL (80% range +20 to +95 mg/dL).")
    assert "about the same" in T.counterfactual_sentence("x", (100, 120), (100, 120), d_peak=0.3)


def test_agreement_on_mock_is_all_true():
    st = mock_data.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    f = backend.predict(st)
    res = T.agreement_check(st, f, backend.predict)
    assert res and all(r.agree for r in res)
    assert {r.key for r in res} <= {k for k, s in T.FEATURE_REGISTRY.items() if s.lever}
