"""Plotly figures. Okabe-Ito palette (colour-blind safe); thresholds dashed; bands translucent;
missing CGM drawn as hatched breaks, never silently interpolated."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.contracts import HYPO_THRESHOLD, SPIKE_THRESHOLD, Forecast, Meal, PatientState
from src.explain import templates as T
from src.serving import timeseries as ts

BLUE, ORANGE, GREEN, VERMILLION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, YELLOW, GREY = "#56B4E9", "#CC79A7", "#F0E442", "#999999"


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _local(x) -> pd.Series | datetime:
    if isinstance(x, datetime):
        return x.astimezone(ts.IST)
    return pd.to_datetime(x, utc=True).dt.tz_convert(ts.IST)


def _layout(fig: go.Figure, height: int, y_range: Sequence[float] | None = None, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="Glucose (mg/dL)",
    )
    if title:
        fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"))
    if y_range:
        fig.update_yaxes(range=list(y_range))
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def _thresholds(fig: go.Figure) -> None:
    fig.add_hline(
        y=SPIKE_THRESHOLD,
        line_dash="dash",
        line_color=VERMILLION,
        line_width=1.5,
        annotation_text="180 spike",
        annotation_position="top left",
    )
    fig.add_hline(
        y=HYPO_THRESHOLD,
        line_dash="dash",
        line_color=PURPLE,
        line_width=1.5,
        annotation_text="70 hypo",
        annotation_position="bottom left",
    )


def _gaps(fig: go.Figure, history: pd.DataFrame, y0: float = 45, y1: float = 380) -> None:
    for i, (a, b) in enumerate(ts.missing_runs(history)):
        a, b = _local(a), _local(b)
        fig.add_trace(
            go.Scatter(
                x=[a, b, b, a, a],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=_rgba(GREY, 0.12),
                fillpattern=dict(shape="/", fgcolor=GREY, size=8),
                name="CGM missing",
                legendgroup="missing",
                showlegend=i == 0,
                hoverinfo="name",
            )
        )


def _meals(
    fig: go.Figure,
    meals: Sequence[Meal],
    y: float = 55,
    color: str = SKY,
    name: str = "Meal (size = GL)",
    now: datetime | None = None,
) -> None:
    if not meals:
        return
    xs = [_local(m.t) for m in meals]
    sizes = [10 + min(m.gl, 90) / 2.5 for m in meals]
    text = [
        f"{m.label}<br>GL {m.gl:.0f} · carbs {m.carbs_g:.0f} g · fibre {m.fibre_g:.0f} g<br>{m.provenance.value}"
        + (" · planned" if now and m.t > now else "")
        for m in meals
    ]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[y] * len(meals),
            mode="markers",
            name=name,
            text=text,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sizes, color=color, symbol="triangle-up", line=dict(width=1, color="#333")),
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
            fillcolor=_rgba(color, alpha),
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
    show_90: bool = True,
    annotate: bool = True,
) -> pd.DataFrame:
    curve = ts.forecast_curve(fc, ts.glucose_now(state.history, state.now))
    x = [_local(state.now + timedelta(minutes=int(m))) for m in curve["t_min"]]
    if show_90 and curve["q05"].notna().all():
        _band(fig, x, curve["q05"], curve["q95"], color, 0.12, f"{label} 90% band", label)
    _band(fig, x, curve["q10"], curve["q90"], color, 0.28, f"{label} 80% band", label)
    suffix = " (interpolated between forecast horizons)" if curve.attrs.get("interpolated") else ""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=curve["q50"],
            mode="lines",
            line=dict(color=color, width=3),
            name=f"{label} median{suffix}",
            legendgroup=label,
            hovertemplate="%{y:.0f} mg/dL<extra>" + label + "</extra>",
        )
    )
    if annotate:
        for t_min, txt in ((ts.spike_crossing(curve), ">180"), (ts.hypo_crossing(curve), "<70")):
            if t_min is not None:
                at = state.now + timedelta(minutes=t_min)
                y = float(curve.loc[curve["t_min"] == t_min, "q50"].iloc[0])
                fig.add_annotation(
                    x=_local(at),
                    y=y,
                    text=f"{txt} at ~{ts.hhmm(at)}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-35 if txt == ">180" else 35,
                    font=dict(size=12),
                )
    return curve


def forecast_figure(
    state: PatientState,
    fc: Forecast,
    *,
    label: str = "Forecast",
    color: str = ORANGE,
    past_hours: float = 4,
    height: int = 420,
    y_range: Sequence[float] | None = None,
    show_meals: bool = True,
    compare: tuple[PatientState, Forecast, str, str] | None = None,
    title: str | None = None,
    show_90: bool = True,
) -> go.Figure:
    fig = go.Figure()
    h = state.history
    t = pd.to_datetime(h["t"], utc=True)
    recent = h[t >= pd.Timestamp(state.now) - pd.Timedelta(hours=past_hours)]
    _gaps(fig, recent)
    fig.add_trace(
        go.Scatter(
            x=_local(recent["t"]),
            y=recent["glucose"],
            mode="lines",
            connectgaps=False,
            line=dict(color=BLUE, width=2.5),
            name="Observed CGM",
            hovertemplate="%{y:.0f} mg/dL<extra>observed</extra>",
        )
    )
    add_forecast(fig, state, fc, color, label, show_90=show_90)
    if compare:
        cstate, cfc, clabel, ccolor = compare
        add_forecast(fig, cstate, cfc, ccolor, clabel, show_90=False, annotate=False)
    if show_meals:
        window_start = state.now - timedelta(hours=past_hours)
        _meals(fig, [m for m in state.meals if m.t >= window_start], now=state.now)
    fig.add_vline(x=_local(state.now), line_dash="dot", line_color=GREY)
    fig.add_annotation(x=_local(state.now), y=1, yref="paper", text="now", showarrow=False, yanchor="bottom")
    _thresholds(fig)
    return _layout(fig, height, y_range, title)


def y_range_for(*pairs: tuple[PatientState, Forecast]) -> tuple[float, float]:
    vals = [60.0, 200.0]
    for state, fc in pairs:
        c = ts.forecast_curve(fc)
        vals += [float(c["q10"].min()), float(c[["q90", "q95"]].max().max())]
        g = state.history["glucose"].dropna()
        vals += [float(g.tail(48).min()), float(g.tail(48).max())] if len(g) else []
    return max(40.0, min(vals) - 10), min(400.0, max(vals) + 15)


def day_figure(history: pd.DataFrame, meals: Sequence[Meal], now: datetime, height: int = 320) -> go.Figure:
    fig = go.Figure()
    _gaps(fig, history)
    fig.add_trace(
        go.Scatter(
            x=_local(history["t"]),
            y=history["glucose"],
            mode="lines",
            connectgaps=False,
            line=dict(color=BLUE, width=2),
            name="Observed CGM",
            hovertemplate="%{y:.0f} mg/dL<extra></extra>",
        )
    )
    _meals(fig, [m for m in meals if m.t <= now], y=55, name="Meal (size = GL)")
    _meals(fig, [m for m in meals if m.t > now], y=55, color=GREY, name="Later meal (after now)", now=now)
    fig.add_vline(x=_local(now), line_dash="dot", line_color=GREY)
    _thresholds(fig)
    return _layout(fig, height)


def attribution_figure(shap: dict[str, float], k: int = 6, height: int = 320) -> go.Figure:
    top = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:k][::-1]
    labels = [f"{T.spec(key).label}  ·  {T.spec(key).group}" for key, _ in top]
    vals = [v for _, v in top]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=[VERMILLION if v > 0 else BLUE for v in vals],
            text=[f"{'+' if v > 0 else '−'}{T.round_mgdl(v)} mg/dL" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f} mg/dL<extra></extra>",
        )
    )
    lo, hi = min([0.0, *vals]), max([0.0, *vals])
    pad = 0.3 * max(hi - lo, 1.0)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[lo - (pad if lo < 0 else 0), hi + pad]),
        xaxis_title="Effect on predicted 2-h peak (mg/dL): red raises ▶, blue lowers ◀",
    )
    fig.add_vline(x=0, line_color=GREY)
    return fig


def attention_figure(
    weights: Sequence[float] | Sequence[Sequence[float]], now: datetime, height: int = 140
) -> go.Figure:
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    n = w.shape[1]
    x = [_local(now - timedelta(minutes=ts.STEP_MIN * (n - 1 - i))) for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=w, x=x, colorscale="Cividis", showscale=False, hovertemplate="%{x|%H:%M}: weight %{z:.3f}<extra></extra>"
        )
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def calibration_figure(df: pd.DataFrame, kind: str, title: str, height: int = 300) -> go.Figure:
    d = df[df["kind"] == kind]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="Perfect"))
    fig.add_trace(
        go.Scatter(
            x=d["nominal"],
            y=d["empirical"],
            mode="lines+markers",
            marker=dict(size=9),
            line=dict(color=BLUE),
            name="Observed",
            hovertemplate="nominal %{x:.2f} → empirical %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title),
        xaxis_title="Nominal / predicted",
        yaxis_title="Empirical",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def clarke_figure(df: pd.DataFrame, height: int = 260) -> go.Figure:
    colors = {"A": GREEN, "B": SKY, "C": YELLOW, "D": ORANGE, "E": VERMILLION}
    fig = go.Figure(
        go.Bar(
            x=df["zone"],
            y=df["share"],
            marker_color=[colors.get(z, GREY) for z in df["zone"]],
            text=[f"{s:.0%}" for s in df["share"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_tickformat=".0%",
        xaxis_title="Clarke Error Grid zone (A+B = clinically acceptable)",
        yaxis_range=[0, 1.05],
    )
    return fig
