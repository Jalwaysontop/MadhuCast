"""Export dashboard charts to docs/figures/ (PNG via kaleido) for the shared slide deck.

    MADHUTWIN_BACKEND=mock python dashboard/export_figures.py

Uses the pinned demo patient and time, so figures match the demo video. Kaleido 1.x needs a
local Chrome/Chromium; if none is found, the script says so and writes HTML instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.components import charts  # noqa: E402
from src.serving import backend, mock_data, scenario  # noqa: E402

OUT = ROOT / "docs" / "figures"


def figures() -> dict:
    state = backend.get_state(mock_data.DEMO_PATIENT, mock_data.DEMO_NOW)
    fc = backend.predict(state)
    idx = len(state.meals) - 1
    s_state, s_fc = backend.run_scenario(state, scenario.preset("Combined", state, idx))
    yr = charts.y_range_for((state, fc), (s_state, s_fc))
    figs = {
        "forecast_P-010": charts.forecast_figure(state, fc, y_range=yr, title="P-010 · 2-hour forecast (MOCK)"),
        "whatif_combined_P-010": charts.forecast_figure(
            state,
            fc,
            label="Baseline",
            y_range=yr,
            compare=(s_state, s_fc, "Rice → millet roti + 20 min walk", charts.GREEN),
            title="What-if: rice → millet roti + 20 min walk (MOCK model estimate)",
        ),
        "attributions_P-010": charts.attribution_figure(fc.shap_values),
    }
    cal, _ = backend.load_report("calibration")
    figs["calibration_coverage"] = charts.calibration_figure(cal, "coverage", "Interval coverage (MOCK)")
    clarke, _ = backend.load_report("clarke_grid")
    figs["clarke_grid"] = charts.clarke_figure(clarke)
    for f in figs.values():
        f.update_layout(
            template="plotly_white",
            width=1100,
            height=600,
            margin=dict(t=70, b=40),
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        )
    return figs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = figures()
    try:
        import kaleido  # noqa: F401

        for name, fig in figs.items():
            fig.write_image(OUT / f"{name}.png", scale=2)
        print(f"wrote {len(figs)} PNGs to {OUT}")
    except Exception as e:  # kaleido missing or no Chrome available
        for name, fig in figs.items():
            fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn")
        print(f"PNG export unavailable ({type(e).__name__}: {e}); wrote HTML to {OUT}")


if __name__ == "__main__":
    main()
