"""The one orchestrated motion: the What-if Twin Curve that morphs the scenario.

`st.plotly_chart` re-mounts on every rerun, which kills Plotly transitions. This
component renders the overlay in a small iframe (Plotly.js from a pinned CDN build),
draws the *previous* scenario first and then `Plotly.animate`s to the new one over
450 ms (ease-out) while the Δ-peak label counts to its new value. With
`prefers-reduced-motion: reduce` it swaps instantly. The previous scenario lives in
st.session_state, so each rerun animates from where the clinician just was.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dashboard.theme.tokens import ALPHA, CHART, FONT_UI, GOOGLE_FONTS, MOTION_MS, Mode, palette, rgba
from src.contracts import HYPO_THRESHOLD, SPIKE_THRESHOLD, Forecast, PatientState
from src.serving import timeseries as ts

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
STATE_KEY = "_twin_morph_prev"


def _x(now, minutes) -> list[str]:
    return [(now + timedelta(minutes=int(m))).astimezone(ts.IST).strftime("%Y-%m-%d %H:%M:%S") for m in minutes]


def _curve(state: PatientState, fc: Forecast) -> dict:
    c = ts.forecast_curve(fc, ts.glucose_now(state.history, state.now))
    return {
        "x": _x(state.now, c["t_min"]),
        "q10": c["q10"].round(1).tolist(),
        "q50": c["q50"].round(1).tolist(),
        "q90": c["q90"].round(1).tolist(),
    }


def payload(
    base_state: PatientState,
    base_fc: Forecast,
    scen_state: PatientState,
    scen_fc: Forecast,
    d_peak: float,
    past_hours: float = 4,
    baseline_only: bool = False,
) -> dict:
    h = base_state.history
    t = pd.to_datetime(h["t"], utc=True)
    recent = h[t >= pd.Timestamp(base_state.now) - pd.Timedelta(hours=past_hours)]
    gy = [None if (m or pd.isna(g)) else round(float(g), 1) for g, m in zip(recent["glucose"], recent["missing_mask"])]
    start = base_state.now - timedelta(hours=past_hours)
    meals = [m for m in base_state.meals if m.t >= start]
    base, scen = _curve(base_state, base_fc), _curve(scen_state, scen_fc)
    i = max(range(len(scen["q50"])), key=lambda k: scen["q50"][k])
    return {
        "history": {
            "x": [
                ts_.astimezone(ts.IST).strftime("%Y-%m-%d %H:%M:%S") for ts_ in pd.to_datetime(recent["t"], utc=True)
            ],
            "y": gy,
        },
        "base": base,
        "scen": scen,
        "peak": {"x": scen["x"][i], "y": scen["q50"][i]},
        "d_peak": round(d_peak, 1),
        "meals": {
            "x": [m.t.astimezone(ts.IST).strftime("%Y-%m-%d %H:%M:%S") for m in meals],
            "size": [8 + min(m.gl, 90) / 4 for m in meals],
            "text": [f"{m.label}, GL {m.gl:.0f}" for m in meals],
        },
        "now": base_state.now.astimezone(ts.IST).strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_only": baseline_only,
    }


def render(data: dict, mode: Mode = "light", height: int = 440, key: str = "twin") -> None:
    base = palette(mode)  # fallback chrome only; live values are read from the parent page's CSS variables
    p = {
        **CHART,
        "ink": base["ink"],
        "ink_muted": base["ink_muted"],
        "hairline": base["hairline"],
        "surface": base["surface"],
    }
    prev = st.session_state.get(f"{STATE_KEY}_{key}")
    start = (
        prev if prev and prev.get("scen", {}).get("x") == data["scen"]["x"] else {"scen": data["base"], "d_peak": 0.0}
    )
    st.session_state[f"{STATE_KEY}_{key}"] = {"scen": data["scen"], "d_peak": data["d_peak"]}
    cfg = {
        "p": p,
        "alpha": ALPHA,
        "target_fill": rgba(p["target"], ALPHA["target_fill"]),
        "band_base": rgba(p["jamun"], 0.14),
        "band_scen": rgba(p["scenario"], ALPHA["scenario_band"]),
        "meal_fill": rgba(p["jamun"], 0.85),
        "spike": SPIKE_THRESHOLD,
        "hypo": HYPO_THRESHOLD,
        "ms": MOTION_MS,
        "font": FONT_UI,
    }
    html = f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{GOOGLE_FONTS}">
<script src="{PLOTLY_JS}"></script>
<style>html,body{{margin:0;background:transparent;font-family:'{FONT_UI}',system-ui,sans-serif}}#c{{width:100%;height:{height}px}}</style>
</head><body><div id="c" role="img" aria-label="Twin curve: baseline forecast versus what-if scenario"></div><script>
const D = {json.dumps(data)}, S = {json.dumps(start)}, C = {json.dumps(cfg)};
const p = C.p, reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = v => (v > 0.4 ? '+' : (v < -0.4 ? '−' : '')) + Math.abs(Math.round(v)) + ' mg/dL';
function scenTraces(s) {{ const v = D.baseline_only ? false : true; return [
  {{x: s.x, y: s.q90, mode:'lines', line:{{width:0}}, hoverinfo:'skip', showlegend:false, visible:v}},
  {{x: s.x, y: s.q10, mode:'lines', line:{{width:0}}, fill:'tonexty', fillcolor:C.band_scen, hoverinfo:'skip', showlegend:false, visible:v}},
  {{x: s.x, y: s.q50, mode:'lines', line:{{color:p.scenario, width:2.5, dash:'dash', shape:'spline'}}, name:'Scenario', visible:v, hovertemplate:'%{{y:.0f}} mg/dL<extra>scenario</extra>'}}]; }}
const b = D.base;
const traces = [
  {{x: D.history.x, y: D.history.y, mode:'lines', connectgaps:false, line:{{color:p.history, width:2}}, name:'Observed CGM', hovertemplate:'%{{y:.0f}} mg/dL<extra>observed</extra>'}},
  {{x: b.x, y: b.q90, mode:'lines', line:{{width:0}}, hoverinfo:'skip', showlegend:false}},
  {{x: b.x, y: b.q10, mode:'lines', line:{{width:0}}, fill:'tonexty', fillcolor:C.band_base, hoverinfo:'skip', showlegend:false}},
  {{x: b.x, y: b.q50, mode:'lines', line:{{color:p.jamun, width:2.5, shape:'spline'}}, name:'Baseline, as eaten', hovertemplate:'%{{y:.0f}} mg/dL<extra>baseline</extra>'}},
  ...scenTraces(S.scen),
  {{x: D.meals.x, y: D.meals.x.map(() => 52), mode:'markers', name:'Meal (size = GL)', text: D.meals.text, hovertemplate:'%{{text}}<extra></extra>',
    marker:{{size: D.meals.size, color: C.meal_fill, line:{{width:0}}}}}}];
const ann = (v) => [
  {{x:1, xref:'paper', y:C.spike, text:'<b>Spike 180</b>', showarrow:false, xanchor:'left', xshift:6, font:{{size:12}}}},
  {{x:1, xref:'paper', y:C.hypo, text:'<b>Hypo 70</b>', showarrow:false, xanchor:'left', xshift:6, font:{{size:12}}}},
  {{x:D.now, y:0, yref:'paper', text:'<b>Now</b>', showarrow:false, xanchor:'left', yanchor:'bottom', xshift:4, yshift:4, font:{{size:12}}}},
  {{x:D.peak.x, y:D.peak.y, text:'<b>Δ peak ' + fmt(v) + '</b>', showarrow:true, arrowhead:0, ax:0, ay:-30, arrowcolor:p.scenario,
    font:{{color:p.scenario, size:13}}, bgcolor:'rgba(0,0,0,0)', borderpad:3, visible: !D.baseline_only}}];
const ys = [].concat(D.history.y.filter(v => v !== null), b.q90, b.q10, D.scen.q90, D.scen.q10, S.scen.q90, S.scen.q10);
const layout = {{
  font:{{family:C.font + ', system-ui, sans-serif', color:p.ink, size:13}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  margin:{{l:44, r:78, t:44, b:30}}, hovermode:'x unified', height:{height},
  hoverlabel:{{bgcolor:p.surface, bordercolor:p.hairline, font:{{color:p.ink, family:C.font}}}},
  legend:{{orientation:'h', y:1.0, yanchor:'bottom', x:0, font:{{color:p.ink_muted, size:12}}, traceorder:'normal'}},
  xaxis:{{type:'date', tickformat:'%H:%M', showgrid:false, linecolor:p.hairline, tickfont:{{color:p.ink_muted, size:12}}, zeroline:false}},
  yaxis:{{range:[Math.max(40, Math.min(...ys) - 12), Math.min(400, Math.max(...ys) + 15)], gridcolor:p.hairline, zeroline:false,
         tickfont:{{color:p.ink_muted, size:12}}, ticksuffix:'  '}},
  shapes:[
    {{type:'rect', xref:'paper', x0:0, x1:1, y0:C.hypo, y1:C.spike, fillcolor:C.target_fill, line:{{width:0}}, layer:'below'}},
    {{type:'line', xref:'paper', x0:0, x1:1, y0:C.spike, y1:C.spike, line:{{color:p.warning, width:1.25, dash:'dash'}}}},
    {{type:'line', xref:'paper', x0:0, x1:1, y0:C.hypo, y1:C.hypo, line:{{color:p.critical, width:1.25, dash:'dash'}}}},
    {{type:'line', yref:'paper', x0:D.now, x1:D.now, y0:0, y1:1, line:{{color:p.history, width:1}}}}],
  annotations: ann(S.d_peak)}};
function liveChrome() {{
  try {{
    const cs = getComputedStyle(window.parent.document.documentElement);
    const v = n => (cs.getPropertyValue(n) || '').trim();
    const ink = v('--mt-ink') || p.ink, muted = v('--mt-ink-muted') || p.ink_muted, hair = v('--mt-hairline') || p.hairline,
          surf = v('--mt-surface') || p.surface;
    return {{ink, muted, hair, surf}};
  }} catch (e) {{ return {{ink:p.ink, muted:p.ink_muted, hair:p.hairline, surf:p.surface}}; }}
}}
function chromeLayout(t) {{ return {{
  'font.color': t.ink, 'legend.font.color': t.muted, 'xaxis.tickfont.color': t.muted, 'yaxis.tickfont.color': t.muted,
  'xaxis.linecolor': t.hair, 'yaxis.gridcolor': t.hair, 'hoverlabel.bgcolor': t.surf, 'hoverlabel.bordercolor': t.hair,
  'hoverlabel.font.color': t.ink }}; }}
let chrome = liveChrome();
Object.assign(layout, {{font:{{family:C.font + ', system-ui, sans-serif', color:chrome.ink, size:13}}}});
Plotly.newPlot('c', traces, layout, {{displayModeBar:false, responsive:true}}).then(() => {{
  Plotly.relayout('c', chromeLayout(chrome));
  setInterval(() => {{ const t = liveChrome(); if (JSON.stringify(t) !== JSON.stringify(chrome)) {{ chrome = t; Plotly.relayout('c', chromeLayout(t)); }} }}, 500);
  const target = scenTraces(D.scen);
  if (reduce) {{ Plotly.animate('c', {{data:target, traces:[4,5,6]}}, {{transition:{{duration:0}}, frame:{{duration:0, redraw:false}}}});
                 Plotly.relayout('c', {{annotations: ann(D.d_peak)}}); return; }}
  Plotly.animate('c', {{data:target, traces:[4,5,6]}}, {{transition:{{duration:C.ms, easing:'cubic-out'}}, frame:{{duration:C.ms, redraw:false}}}});
  const t0 = performance.now(), from = S.d_peak, to = D.d_peak;
  function tick(t) {{ const k = Math.min(1, (t - t0) / C.ms), e = 1 - Math.pow(1 - k, 3);
    Plotly.relayout('c', {{annotations: ann(from + (to - from) * e)}}); if (k < 1) requestAnimationFrame(tick); }}
  requestAnimationFrame(tick);
}});
</script></body></html>"""
    components.html(html, height=height + 8, scrolling=False)
