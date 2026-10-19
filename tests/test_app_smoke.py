"""Streamlit AppTest smoke tests: every page renders on the mock backend without exceptions."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = {
    "pages/alert_queue.py": "Alert queue",
    "pages/patient_card.py": "Patient P-010",
    "pages/forecast_view.py": "Forecast",
    "pages/meal_timeline.py": "Meals",
    "pages/whatif_lab.py": "What-if lab",
    "pages/explain_panel.py": "Why this forecast?",
    "pages/trust_page.py": "Model trust",
}

pytestmark = pytest.mark.smoke


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("MADHUTWIN_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("MADHUTWIN_DEMO", "1")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at) -> str:
    return " ".join(str(e.value) for e in (*at.title, *at.markdown, *at.caption, *at.warning, *at.info))


@pytest.mark.parametrize("page,title", PAGES.items())
def test_page_renders(app, page, title):
    app.switch_page(page).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert any(t.value == title for t in app.title)


def test_global_chrome_on_full_app(app):
    # AppTest.switch_page runs the page script alone, so app.py's chrome is checked on the full-app run.
    text = _texts(app)
    assert "Synthetic/open data only. Research PoC, not a medical device." in text
    assert "<b>Patient</b> P-010" in text and "<b>Backend</b> MOCK" in text
    assert any("Backend · data **MOCK**" in c.value for c in app.sidebar.caption)


def test_alert_queue_content(app):
    text = _texts(app)
    assert "UNCALIBRATED" in text
    assert any(m.label.startswith("Alerts / patient-day") for m in app.metric)
    labels = [e.label for e in app.expander]
    assert any(lab.startswith("P-011 · hypo · CRITICAL") for lab in labels)
    assert any(lab.startswith("P-007 · unknown") for lab in labels)


def test_whatif_preset_changes_scenario(app):
    app.switch_page("pages/whatif_lab.py").run()
    assert "not a proven causal effect" in _texts(app)
    app.button(key="preset-Combined").click().run()
    assert not app.exception
    delta = next(m for m in app.metric if m.label == "Δ peak")
    assert delta.value.startswith("-")
    assert any("Compared with the scenario" in str(m.value) for m in app.markdown)


def test_feedback_and_audit_logging(app, tmp_path):
    from src.serving import logbook

    btn = next(b for b in app.button if b.key and b.key.startswith("fb-") and b.key.endswith("-useful"))
    btn.click().run()
    fb = logbook.read("feedback.jsonl")
    assert fb and fb[-1]["label"] == "useful" and fb[-1]["model_version"]
    app.switch_page("pages/forecast_view.py").run()
    app.button(key="fv-health-pull").click().run()
    audit = logbook.read("audit.jsonl")
    assert audit and audit[-1]["action"] == "request_raw_window" and audit[-1]["stub"] is True


def test_patient_switch_and_unknown_state(app):
    sb = app.selectbox(key="pid_select")
    sb.select(next(o for o in sb.options if o.startswith("P-007"))).run()
    app.switch_page("pages/forecast_view.py").run()
    assert not app.exception
    assert any(m.label == "Current glucose" and m.value == "UNKNOWN" for m in app.metric)
