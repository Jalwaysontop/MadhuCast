"""Streamlit AppTest smoke tests: every page renders on the mock backend without exceptions."""

import re
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


def readout(at, label: str) -> str | None:
    """Value of a kit readout (rendered with data-label / data-value attributes)."""
    pat = re.compile(r'data-label="' + re.escape(label) + r'" data-value="([^"]*)"')
    for m in at.markdown:
        hit = pat.search(str(m.value))
        if hit:
            return hit.group(1)
    return None


@pytest.mark.parametrize("page,title", PAGES.items())
def test_page_renders(app, page, title):
    app.switch_page(page).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert any(t.value == title for t in app.title)


def test_global_chrome_on_full_app(app):
    # AppTest.switch_page runs the page script alone, so app.py's chrome is checked on the full-app run.
    text = _texts(app)
    assert "Research PoC" not in text and "Backend" not in text
    assert 'class="mt-pid">P-010' in text
    assert all(b in text for b in ('data-prov="REAL"', 'data-prov="SYNTHETIC"', 'data-prov="MODEL_ESTIMATE"'))


def test_alert_queue_content(app):
    text = _texts(app)
    assert "Default thresholds" in text
    assert float(readout(app, "Alert burden")) <= 4
    assert 'data-tier="CRITICAL"' in text and 'class="critical"' in text  # critical row carries the left rule
    labels = [e.label for e in app.expander]
    assert any(lab.startswith("P-011: hypo below 70, Critical") for lab in labels)
    assert any(lab.startswith("P-007: data stale, Unknown") for lab in labels)


def test_whatif_preset_changes_scenario(app):
    app.switch_page("pages/whatif_lab.py").run()
    assert "not a proven causal effect" in _texts(app)
    app.button(key="preset-Combined").click().run()
    assert not app.exception
    assert readout(app, "Δ peak").startswith("−")
    assert readout(app, "Δ time above 180").startswith("−")
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
    assert readout(app, "Current glucose") == "UNKNOWN"
    assert 'data-tier="UNKNOWN"' in _texts(app)  # stale stream is loud, never "low risk"


@pytest.mark.parametrize("data", ["mock", "real"])
def test_pages_render_on_real_model(data, monkeypatch, tmp_path, request):
    """P2's hybrid XGBoost adapter (and P3's exported data): every page renders, and the hybrid is disclosed."""
    if data == "real":
        request.getfixturevalue("real_data")
    monkeypatch.setenv("MADHUTWIN_MODEL", "real")
    monkeypatch.setenv("MADHUTWIN_LOG_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception and not at.error, (at.exception, [e.value for e in at.error])
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception and not at.error, (page, at.exception, [e.value for e in at.error])
    at.switch_page("pages/whatif_lab.py").run()
    at.button(key="preset-+20 min post-meal walk").click().run()
    assert readout(at, "Δ spike probability") == "n/a"
    assert any("cannot see a post-meal walk" in str(m.value) for m in at.markdown)
