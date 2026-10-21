"""Plotly figures in the Jamun system. The hero is the Twin Curve: ink history flowing
into a translucent jamun forecast cone over the shaded 70–180 mg/dL target band.

Severity colours carry meaning only; thresholds are dashed with right-edge labels;
missing CGM is a hatched UNKNOWN region, never interpolated.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import plotly_template
from dashboard.theme.tokens import ALPHA, CHART, FOOD_HINDI, Mode, rgba
from src.contracts import HYPO_THRESHOLD, SPIKE_THRESHOLD, Forecast, Meal, PatientState
from src.explain import templates as T
from src.serving import timeseries as ts

# Back-compat names (pages/export use charts.GREEN for the scenario colour)
GREEN = "scenario"


def palette(mode: Mode = "light") -> dict[str, str]:
    """Chart marks are theme-proof mid-tones; surface/ink only feed hover fallbacks."""
    return {
        **CHART,
        "ink": CHART["history"],
        "surface": "rgba(0,0,0,0)",
        "hairline": "rgba(128,128,128,0.25)",
        "ink_muted": "#7A8294",
    }


def _local(x) -> pd.Series | datetime:
    if isinstance(x, datetime):
        return x.astimezone(ts.IST)
    return pd.to_datetime(x, utc=True).dt.tz_convert(ts.IST)


def _base(mode: Mode, height: int, y_range: Sequence[float] | None, title: str | None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=76, t=58 if title else 36, b=8),
        yaxis_title=None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, itemwidth=30),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.99, yanchor="top", font=dict(size=14)))
    if y_range:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_yaxes(ticksuffix="  ")
    return fig


def _target_and_thresholds(fig: go.Figure, p: dict[str, str]) -> None:
    fig.add_hrect(
        y0=HYPO_THRESHOLD,
        y1=SPIKE_THRESHOLD,
        line_width=0,
        layer="below",
        fillcolor=rgba(p["target"], ALPHA["target_fill"]),
    )
    for y, color, label in ((SPIKE_THRESHOLD, p["warning"], "Spike 180"), (HYPO_THRESHOLD, p["critical"], "Hypo 70")):
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1.25)
        fig.add_annotation(
            x=1,
            xref="paper",
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=12),
        )


def _now(fig: go.Figure, now: datetime, p: dict[str, str]) -> None:
    fig.add_vline(x=_local(now), line_color=CHART["history"], line_width=1)
    fig.add_annotation(
        x=_local(now),
        y=1,
        yref="paper",
        text="<b>Now</b>",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=4,
        font=dict(size=12),
    )


def _gaps(fig: go.Figure, history: pd.DataFrame, p: dict[str, str], y0: float = 45, y1: float = 380) -> None:
    for i, (a, b) in enumerate(ts.missing_runs(history)):
        a, b = _local(a), _local(b)
        fig.add_trace(
            go.Scatter(
                x=[a, b, b, a, a],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=rgba(p["unknown"], 0.10),
                fillpattern=dict(shape="/", fgcolor=p["unknown"], size=8, solidity=0.15),
                name="CGM missing (unknown)",
                legendgroup="missing",
                showlegend=i == 0,
                hoverinfo="name",
            )
        )


def _meals(
    fig: go.Figure, meals: Sequence[Meal], p: dict[str, str], now: datetime | None = None, y: float = 52
) -> None:
    if not meals:
        return
    text = []
    for m in meals:
        items = ", ".join(
            f"{FOOD_HINDI.get(i['food'], '')} {i['food']} ×{i['portion']:g}".strip() for i in (m.items or [])
        )
        text.append(
            f"<b>{m.label}</b><br>{items}<br>GL {m.gl:.0f} · carbs {m.carbs_g:.0f} g · fibre {m.fibre_g:.0f} g"
            + (" · planned" if now and m.t > now else "")
            + f"<br>{m.provenance.value}"
        )
    fig.add_trace(
        go.Scatter(
            x=[_local(m.t) for m in meals],
            y=[y] * len(meals),
            mode="markers",
            name="Meal (size = GL)",
            text=text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=[8 + min(m.gl, 90) / 4 for m in meals],
                color=rgba(p["jamun"], 0.85),
                line=dict(width=1.5, color=p["surface"]),
            ),
        )
    )


def _band(fig: go.Figure, x, lo, hi, color: str, alpha: float, name: str, group: str, show: bool = True) -> None:
    fig.add_trace(
        go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, legendgroup=group, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lo,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(color, alpha),
            name=name,
            legendgroup=group,
            showlegend=show,
            hoverinfo="skip",
        )
    )


def add_forecast(
    fig: go.Figure,
    state: PatientState,
    fc: Forecast,
    color: str,
    label: str,
    p: dict[str, str],
    show_90: bool = True,
    annotate: bool = True,
    dashed: bool = False,
) -> pd.DataFrame:
    curve = ts.forecast_curve(fc, ts.glucose_now(state.history, state.now))
    x = [_local(state.now + timedelta(minutes=int(m))) for m in curve["t_min"]]
    if show_90 and curve["q05"].notna().all():
        _band(fig, x, curve["q05"], curve["q95"], color, ALPHA["band90"], f"{label} 90% range", label, show=False)
    _band(
        fig,
        x,
        curve["q10"],
        curve["q90"],
        color,
        ALPHA["band80"] if not dashed else ALPHA["scenario_band"],
        "80% and 90% range" if show_90 else "80% range",
        label,
    )
    suffix = " (interpolated between horizons)" if curve.attrs.get("interpolated") else ""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=curve["q50"],
            mode="lines",
            name=f"{label}{suffix}",
            legendgroup=label,
            line=dict(color=color, width=2.5, dash="dash" if dashed else "solid", shape="spline"),
            hovertemplate="%{y:.0f} mg/dL<extra>" + label + "</extra>",
        )
    )
    if annotate:
        for t_min, txt, c in (
            (ts.spike_crossing(curve), "Spike", p["warning"]),
            (ts.hypo_crossing(curve), "Hypo", p["critical"]),
        ):
            if t_min:
                at = _local(state.now + timedelta(minutes=t_min))
                yv = SPIKE_THRESHOLD if txt == "Spike" else HYPO_THRESHOLD
                fig.add_trace(
                    go.Scatter(
                        x=[at],
                        y=[yv],
                        mode="markers",
                        showlegend=False,
                        hoverinfo="skip",
                        marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=c, width=2.5)),
                    )
                )
                fig.add_annotation(
                    x=at,
                    y=yv,
                    text=f"<b>{txt} ~{t_min} min</b>",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom" if txt == "Spike" else "top",
                    xshift=8,
                    yshift=6 if txt == "Spike" else -6,
                    font=dict(size=12),
                )
    return curve


def forecast_figure(
    state: PatientState,
    fc: Forecast,
    *,
    label: str = "Twin forecast",
    color: str = "jamun",
    past_hours: float = 4,
    height: int = 420,
    y_range: Sequence[float] | None = None,
    show_meals: bool = True,
    compare: tuple[PatientState, Forecast, str, str] | None = None,
    title: str | None = None,
    show_90: bool = True,
    mode: Mode = "light",
) -> go.Figure:
    """The Twin Curve. ``color``/compare colours are token names ("jamun", "scenario")."""
    p = palette(mode)
    fig = _base(mode, height, y_range, title)
    _target_and_thresholds(fig, p)
    h = state.history
    t = pd.to_datetime(h["t"], utc=True)
    recent = h[t >= pd.Timestamp(state.now) - pd.Timedelta(hours=past_hours)]
    _gaps(fig, recent, p)
    fig.add_trace(
        go.Scatter(
            x=_local(recent["t"]),
            y=recent["glucose"],
            mode="lines",
            connectgaps=False,
            line=dict(color=p["ink"], width=2),
            name="Observed CGM",
            hovertemplate="%{y:.0f} mg/dL<extra>observed</extra>",
        )
    )
    add_forecast(fig, state, fc, p.get(color, color), label, p, show_90=show_90, dashed=color == "scenario")
    if compare:
        cstate, cfc, clabel, ccolor = compare
        add_forecast(
            fig,
            cstate,
            cfc,
            p.get(ccolor, ccolor),
            clabel,
            p,
            show_90=False,
            annotate=False,
            dashed=ccolor == "scenario",
        )
    if show_meals:
        start = state.now - timedelta(hours=past_hours)
        _meals(fig, [m for m in state.meals if m.t >= start], p, now=state.now, y=(y_range[0] + 9) if y_range else 52)
    _now(fig, state.now, p)
    return fig


def y_range_for(*pairs: tuple[PatientState, Forecast]) -> tuple[float, float]:
    vals = [60.0, 200.0]
    for state, fc in pairs:
        c = ts.forecast_curve(fc)
        vals += [float(c["q10"].min()), float(c[["q90", "q95"]].max().max())]
        g = state.history["glucose"].dropna()
        vals += [float(g.tail(48).min()), float(g.tail(48).max())] if len(g) else []
    return max(40.0, min(vals) - 12), min(400.0, max(vals) + 15)


def day_figure(
    history: pd.DataFrame, meals: Sequence[Meal], now: datetime, height: int = 300, mode: Mode = "light"
) -> go.Figure:
    p = palette(mode)
    fig = _base(mode, height, None, None)
    _target_and_thresholds(fig, p)
    _gaps(fig, history, p)
    fig.add_trace(
        go.Scatter(
            x=_local(history["t"]),
            y=history["glucose"],
            mode="lines",
            connectgaps=False,
            line=dict(color=p["ink"], width=1.75),
            name="Observed CGM",
            hovertemplate="%{y:.0f} mg/dL<extra></extra>",
        )
    )
    _meals(fig, [m for m in meals if m.t <= now], p)
    later = [m for m in meals if m.t > now]
    if later:
        fig.add_trace(
            go.Scatter(
                x=[_local(m.t) for m in later],
                y=[52] * len(later),
                mode="markers",
                name="Later meal (after now)",
                hovertemplate="%{x|%H:%M} planned<extra></extra>",
                marker=dict(
                    size=[8 + min(m.gl, 90) / 4 for m in later],
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.5, color=p["unknown"]),
                ),
            )
        )
    _now(fig, now, p)
    return fig


def attribution_figure(shap: dict[str, float], k: int = 6, height: int = 320, mode: Mode = "light") -> go.Figure:
    p = palette(mode)
    top = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:k][::-1]
    vals = [v for _, v in top]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[T.spec(key).label for key, _ in top],
            orientation="h",
            marker_color=[p["warning"] if v > 0 else p["target"] for v in vals],
            text=[f"{'+' if v > 0 else '−'}{T.round_mgdl(v)}" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f} mg/dL<extra></extra>",
        )
    )
    lo, hi = min([0.0, *vals]), max([0.0, *vals])
    pad = 0.3 * max(hi - lo, 1.0)
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        showlegend=False,
        xaxis=dict(range=[lo - (pad if lo < 0 else 0), hi + pad], title="mg/dL change in predicted peak"),
    )
    fig.add_vline(x=0, line_color=p["hairline"])
    return fig


def attention_figure(
    weights: Sequence[float] | Sequence[Sequence[float]], now: datetime, height: int = 96, mode: Mode = "light"
) -> go.Figure:
    p = palette(mode)
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    n = w.shape[1]
    x = [_local(now - timedelta(minutes=ts.STEP_MIN * (n - 1 - i))) for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=w,
            x=x,
            colorscale=[[0, rgba(p["jamun"], 0.06)], [1, p["jamun"]]],
            showscale=False,
            hovertemplate="%{x|%H:%M}: weight %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=4, b=4),
    )
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(tickformat="%H:%M", showline=False)
    return fig


def calibration_figure(df: pd.DataFrame, kind: str, title: str, height: int = 300, mode: Mode = "light") -> go.Figure:
    p = palette(mode)
    d = df[df["kind"] == kind]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color=p["unknown"], width=1),
            name="Perfect calibration",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["nominal"],
            y=d["empirical"],
            mode="lines+markers",
            marker=dict(size=8, color=p["jamun"]),
            line=dict(color=p["jamun"], width=2),
            name="Observed",
            hovertemplate="nominal %{x:.2f} → empirical %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        title=dict(text=title, x=0, font=dict(size=14)),
        xaxis_title="Nominal / predicted",
        yaxis_title="Empirical",
        margin=dict(l=8, r=8, t=56, b=8),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def ablation_dots(
    df: pd.DataFrame, metric: str, height: int = 260, mode: Mode = "light", highlight: str | None = None
) -> go.Figure:
    """Dot plot with CI whiskers: one row per variant."""
    p = palette(mode)
    d = df[df["metric"] == metric].iloc[::-1]
    colors = [p["jamun"] if v == highlight else p["ink_muted"] for v in d["variant"]]
    fig = go.Figure(
        go.Scatter(
            x=d["value"],
            y=d["variant"],
            mode="markers",
            marker=dict(size=10, color=colors),
            error_x=dict(
                type="data",
                symmetric=False,
                array=d["ci_high"] - d["value"],
                arrayminus=d["value"] - d["ci_low"],
                color=p["ink_muted"],
                thickness=1.5,
                width=4,
            ),
            hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        showlegend=False,
        title=dict(text=metric.replace("_", " "), x=0, font=dict(size=14)),
        margin=dict(l=8, r=8, t=40, b=8),
    )
    fig.update_xaxes(showgrid=True, gridcolor=p["hairline"])
    fig.update_yaxes(showgrid=False)
    return fig


def clarke_figure(df: pd.DataFrame, height: int = 240, mode: Mode = "light") -> go.Figure:
    p = palette(mode)
    colors = {"A": p["target"], "B": rgba(p["target"], 0.55), "C": p["warning"], "D": p["critical"], "E": p["critical"]}
    fig = go.Figure(
        go.Bar(
            x=df["zone"],
            y=df["share"],
            marker_color=[colors.get(z, p["unknown"]) for z in df["zone"]],
            text=[f"{s:.0%}" for s in df["share"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        template=plotly_template.name(mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        showlegend=False,
        yaxis_tickformat=".0%",
        xaxis_title="Clarke Error Grid zone (A + B clinically acceptable)",
        yaxis_range=[0, 1.08],
    )
    return fig
