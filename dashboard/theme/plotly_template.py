"""Plotly templates `madhutwin_light` / `madhutwin_dark`, built from tokens.py.

Hanken Grotesk, muted axes, y-only hairline grid, no top/right spines, surface
hover labels with a hairline border. Chart-specific decorations (target band,
thresholds, "Now") are added by the Twin Curve builder, not the template.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from dashboard.theme.tokens import FONT_UI, Mode, palette


def _template(mode: Mode) -> go.layout.Template:
    p = palette(mode)
    axis = dict(
        showline=True,
        linecolor=p["hairline"],
        linewidth=1,
        mirror=False,
        ticks="outside",
        tickcolor=p["hairline"],
        ticklen=4,
        tickfont=dict(color=p["ink_muted"], size=12),
        title=dict(font=dict(color=p["ink_muted"], size=12)),
        zeroline=False,
        showgrid=False,
        automargin=True,
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family=f"{FONT_UI}, system-ui, sans-serif", color=p["ink"], size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[p["jamun"], p["scenario"], p["target"], p["watch"], p["warning"], p["unknown"]],
            xaxis=axis,
            yaxis={
                **axis,
                "showgrid": True,
                "gridcolor": p["hairline"],
                "gridwidth": 1,
                "showline": False,
                "ticks": "",
            },
            hoverlabel=dict(
                bgcolor=p["surface"], bordercolor=p["hairline"], font=dict(color=p["ink"], family=FONT_UI, size=13)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                font=dict(color=p["ink_muted"], size=12),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=8, r=8, t=28, b=8),
            hovermode="x unified",
        )
    )


def register() -> None:
    for mode in ("light", "dark"):
        pio.templates[f"madhutwin_{mode}"] = _template(mode)  # type: ignore[arg-type]
    pio.templates.default = "madhutwin_light"


def name(mode: Mode) -> str:
    if "madhutwin_light" not in pio.templates:
        register()
    return f"madhutwin_{mode}"
