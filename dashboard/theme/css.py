"""One injected stylesheet, generated from tokens.py.

Targets stable hooks only: CSS variables on :root, our own `mt-*` classes, the
`st-key-<key>` class Streamlit emits for keyed containers, and `data-testid`
attributes. Never auto-generated `css-xxxx` / emotion classes (they change on upgrade).
Selectors stay one class deep so rules don't fight each other.
"""

from __future__ import annotations

from dashboard.theme.tokens import ALPHA, FONT_DEVANAGARI, FONT_UI, GOOGLE_FONTS, MOTION_MS, Mode, palette, rgba


def _vars(mode: Mode) -> str:
    p = palette(mode)
    lines = [f"--mt-{k.replace('_', '-')}: {v};" for k, v in p.items()]
    for k in ("critical", "warning", "watch", "unknown", "target", "jamun"):
        lines.append(f"--mt-{k}-tint: {rgba(p[k], ALPHA['tint'])};")
    lines.append(f"--mt-shadow-tray: 0 8px 24px -4px {rgba('#1B2130', 0.10 if mode == 'light' else 0.45)};")
    return "\n  ".join(lines)


THEME_SYNC_JS = """
<script>
(function () {
  const d = window.parent && window.parent.document; if (!d) return;
  function sync() {
    const app = d.querySelector('[data-testid="stApp"]') || d.body;
    const m = (getComputedStyle(app).backgroundColor || '').match(/\\d+/g); if (!m) return;
    const lum = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
    const mode = lum < 128 ? 'dark' : 'light';
    if (d.documentElement.dataset.mtTheme !== mode) d.documentElement.dataset.mtTheme = mode;
  }
  sync(); setInterval(sync, 500);
})();
</script>
"""


def stylesheet(mode: Mode = "light") -> str:
    """Both palettes are emitted: `mode` (server-detected) is the default, and
    `:root[data-mt-theme=…]` (set by THEME_SYNC_JS from the live app background) wins,
    so switching theme in Streamlit's menu restyles everything without a rerun."""
    return f"""
@import url('{GOOGLE_FONTS}');
:root[data-mt-theme="light"] {{ {_vars("light")} }}
:root[data-mt-theme="dark"] {{ {_vars("dark")} }}
:root {{
  {_vars(mode)}
  --mt-font: '{FONT_UI}', system-ui, sans-serif;
  --mt-font-deva: '{FONT_DEVANAGARI}', '{FONT_UI}', sans-serif;
  --mt-motion: {MOTION_MS}ms;
}}
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--mt-font); }}
[data-testid="stMain"] .block-container {{ padding-top: 3.75rem; max-width: 1360px; }}
.mt-num, [data-testid="stMetricValue"], [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
*:focus-visible {{ outline: 2px solid var(--mt-jamun) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ transition: none !important; animation: none !important; }} }}

/* ---------- Plotly chrome follows the live theme (marks use theme-proof mid-tones) */
.js-plotly-plot .main-svg text {{ fill: var(--mt-ink-muted) !important; }}
.js-plotly-plot .annotation-text {{ fill: var(--mt-ink) !important; font-weight: 600; }}
.js-plotly-plot .gridlayer path, .js-plotly-plot .zerolinelayer path {{ stroke: var(--mt-hairline) !important; }}
.js-plotly-plot .xlines-above, .js-plotly-plot .ylines-above, .js-plotly-plot path.xtick, .js-plotly-plot path.ytick {{ stroke: var(--mt-hairline) !important; }}
.js-plotly-plot .hoverlayer .bg {{ fill: var(--mt-surface) !important; stroke: var(--mt-hairline) !important; }}
.js-plotly-plot .hoverlayer text {{ fill: var(--mt-ink) !important; }}
.js-plotly-plot .bglayer .bg {{ fill: transparent !important; }}

/* ---------- sidebar navigation */
[data-testid="stSidebarNavLink"] {{ border-radius: 8px; color: var(--mt-ink-muted); font-weight: 500; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: var(--mt-jamun-tint); color: var(--mt-jamun); font-weight: 600; }}

/* ---------- panels (keyed containers: st.container(key="panel-…")) */
[class*="st-key-panel"] {{ background: var(--mt-surface); border: 1px solid var(--mt-hairline); border-radius: 8px; padding: 20px 24px; }}
.st-key-lever-tray {{ background: var(--mt-surface); border: 1px solid var(--mt-hairline); border-radius: 8px; padding: 20px; box-shadow: var(--mt-shadow-tray); }}
[class*="st-key-quiet"] {{ color: var(--mt-ink-muted); }}
.st-key-theme-sync {{ position: absolute; height: 0; overflow: hidden; }}
.st-key-lever-tray button p {{ font-size: 13px; white-space: nowrap; }}
.st-key-lever-tray [data-testid="stDataFrame"] {{ font-family: var(--mt-font-deva); }}

/* ---------- header bar */
.mt-header {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 12px 20px; background: var(--mt-surface);
  border: 1px solid var(--mt-hairline); border-radius: 8px; margin-bottom: 4px; font-size: 14px; color: var(--mt-ink-muted); }}
.mt-header b {{ color: var(--mt-ink); font-weight: 600; }}
.mt-header .mt-pid {{ color: var(--mt-ink); font-weight: 700; font-size: 16px; }}
.mt-header .mt-spacer {{ flex: 1; }}
.mt-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 14px; }}
.mt-meter {{ display: inline-flex; align-items: flex-end; gap: 2px; height: 15px; vertical-align: -2px; }}
.mt-meter i {{ display: block; width: 4px; border-radius: 1px; background: var(--mt-hairline); }}
.mt-meter i.on {{ background: var(--mt-target); }}
.mt-meter.warn i.on {{ background: var(--mt-warning); }}
.mt-meter.bad i.on {{ background: var(--mt-critical); }}

/* ---------- badges */
.mt-badge {{ display: inline-flex; align-items: center; gap: 4px; padding: 1px 6px; border-radius: 2px; border: 1px solid currentColor;
  font-size: 12px; font-weight: 500; line-height: 16px; white-space: nowrap; vertical-align: middle; background: var(--mt-surface); }}
.mt-badge.mock {{ border-style: dashed; }}
.mt-tier {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px 3px 8px; border-radius: 2px; font-size: 14px;
  font-weight: 600; line-height: 20px; white-space: nowrap; vertical-align: middle; }}
.mt-tier.unknown {{ border: 1px dashed currentColor; }}
.mt-icon {{ flex: none; }}

/* ---------- readouts */
.mt-readouts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 16px; }}
.mt-readout {{ background: var(--mt-surface); border: 1px solid var(--mt-hairline); border-radius: 8px; padding: 16px 20px; min-width: 0; }}
.mt-readout.alert {{ border-left: 3px solid var(--mt-warning); }}
.mt-readout.alert.critical {{ border-left-color: var(--mt-critical); }}
.mt-readout .mt-r-label {{ font-size: 14px; color: var(--mt-ink-muted); display: flex; gap: 6px; align-items: center; }}
.mt-readout .mt-r-row {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0 6px; margin: 2px 0; }}
.mt-readout .mt-r-value {{ font-size: 39px; font-weight: 700; letter-spacing: -0.02em; line-height: 44px; color: var(--mt-ink); font-variant-numeric: tabular-nums; }}
.mt-readout .mt-r-value.good {{ color: var(--mt-target); }}
.mt-readout .mt-r-value.bad {{ color: var(--mt-critical); }}
.mt-readout .mt-r-unit {{ font-size: 14px; font-weight: 500; color: var(--mt-ink-muted); white-space: nowrap; }}
.mt-readout .mt-r-foot {{ font-size: 12px; font-weight: 500; color: var(--mt-ink-muted); }}
.mt-readout .mt-r-foot.warn {{ color: var(--mt-warning); font-weight: 600; }}
.mt-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: var(--mt-surface); border: 1px solid var(--mt-hairline); border-radius: 8px; }}
.mt-strip .mt-readout {{ border: 0; border-radius: 0; background: transparent; }}
.mt-strip .mt-readout + .mt-readout {{ border-left: 1px solid var(--mt-hairline); }}

/* ---------- definition list (patient record) */
.mt-dl {{ margin: 0; }}
.mt-dl > div {{ padding: 9px 0; border-top: 1px solid var(--mt-hairline); }}
.mt-dl dt {{ font-size: 12px; font-weight: 500; color: var(--mt-ink-muted); }}
.mt-dl dd {{ margin: 2px 0 0; font-size: 14px; font-weight: 600; color: var(--mt-ink); display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

/* ---------- triage table */
.mt-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--mt-surface); border: 1px solid var(--mt-hairline);
  border-radius: 8px; overflow: hidden; font-size: 14px; color: var(--mt-ink); font-variant-numeric: tabular-nums; }}
.mt-table th {{ white-space: nowrap; background: var(--mt-sunken); color: var(--mt-ink-muted); font-size: 12px; font-weight: 500; text-align: left; padding: 10px 10px; }}
.mt-table-wrap {{ overflow-x: auto; border-radius: 8px; }}
.mt-table td {{ white-space: nowrap; padding: 11px 10px; border-top: 1px solid var(--mt-hairline); vertical-align: middle; }}
.mt-table tr.critical td:first-child {{ box-shadow: inset 3px 0 0 var(--mt-critical); }}
.mt-table .muted {{ color: var(--mt-ink-muted); font-size: 12px; margin-left: 6px; }}
.mt-table .low {{ color: var(--mt-watch); }}
.mt-table .warn {{ color: var(--mt-warning); }}
.mt-bar {{ display: inline-block; width: 56px; height: 6px; border-radius: 3px; background: var(--mt-sunken); vertical-align: middle; margin-right: 8px; overflow: hidden; }}
.mt-bar i {{ display: block; height: 100%; border-radius: 3px; }}

/* ---------- prose, disclaimers, empty states */
.mt-lead {{ font-size: 20px; line-height: 30px; color: var(--mt-ink); max-width: 72ch; margin: 0 0 4px; }}
.mt-prose {{ max-width: 78ch; }}
.mt-muted {{ color: var(--mt-ink-muted); font-size: 14px; }}
.mt-caption {{ color: var(--mt-ink-muted); font-size: 12px; font-weight: 500; }}
.mt-disclaimer {{ display: flex; gap: 10px; align-items: flex-start; background: var(--mt-jamun-tint); color: var(--mt-ink); border-radius: 8px;
  padding: 10px 14px; font-size: 14px; line-height: 20px; max-width: 100%; }}
.mt-empty {{ border: 1px dashed var(--mt-hairline); border-radius: 8px; padding: 20px 24px; color: var(--mt-ink); background: var(--mt-surface); }}
.mt-empty b {{ display: block; margin-bottom: 4px; }}
.mt-chip {{ display: inline-block; padding: 0 7px; margin: 0 4px 2px 0; border-radius: 2px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--mt-hairline); color: var(--mt-ink); font-variant-numeric: tabular-nums; }}
.mt-deva {{ font-family: var(--mt-font-deva); font-weight: 500; }}
.mt-footer {{ font-size: 12px; color: var(--mt-ink-muted); line-height: 2; display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }}
.mt-footer b {{ color: var(--mt-ink); font-weight: 600; }}
.mt-group {{ font-size: 12px; font-weight: 600; color: var(--mt-ink-muted); margin: 10px 0 2px; }}
.mt-check {{ display: flex; gap: 10px; align-items: flex-start; padding: 6px 0; }}
.mt-check b {{ display: block; font-size: 14px; }}
.mt-scrubber-title {{ font-size: 14px; font-weight: 600; color: var(--mt-ink); }}
.mt-motion {{ transition: all var(--mt-motion) cubic-bezier(.2,.8,.2,1); }}
"""
