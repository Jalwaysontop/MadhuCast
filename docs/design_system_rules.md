# Jamun design-system rules (for humans and agents)

Follow these whenever you touch `dashboard/`. The Figma file "MadhuTwin — Jamun UI" and `dashboard/theme/tokens.py` are the sources of truth. Accessibility rules beat visual values.

1. **Colour comes from tokens only.**
   - Page styling: `var(--mt-<token>)`, emitted by `dashboard/theme/css.py`.
   - Chart marks: `tokens.CHART`, which is theme-proof.
   - Never hard-code hex values in pages.
2. **Colour means something.**
   - jamun = the twin, forecast and primary action.
   - target = the 70–180 range.
   - critical / warning / watch / unknown = severity.
   - scenario = the what-if curve, always dashed.
   - No decorative colour.
3. **Severity is never colour alone.** Always use `kit.tier_badge` (icon + word + colour). Unknown is dashed or hatched.
4. **Provenance is always visible.** Every number shown has a `kit.provenance_badge` nearby (REAL solid, SYNTHETIC flask, MOCK dashed, MODEL ESTIMATE jamun). Never hide one, shrink it below 12 px, or mute it below AA.
5. **Pages compose the kit; they hold no raw HTML or CSS.** Use `dashboard/components/kit.py`: `header_bar`, `Readout`/`readouts`, `delta_strip`, `alert_card`, `triage_table`, `driver_bars`, `definition_list`, `disclaimer`, `empty_state`, `lead`, `caption`. Add new components to the kit.
6. **Use only stable selectors.**
   - Allowed: keyed containers (`st.container(key="panel-…")` → `.st-key-panel-…`), `mt-*` classes, `data-testid`.
   - Never auto-generated `css-xxxx` or emotion classes.
   - Keep selectors one class deep.
7. **Numbers.**
   - Tabular numerals.
   - Big readouts are 39/700 with the unit at 14/500 *beside* the value.
   - Glucose is rounded to the nearest 5 above 10.
   - Use a typographic minus (−).
8. **Words.**
   - Sentence case. No ALL-CAPS labels, no `A · B · C` strings, no `→` appended to buttons, no emoji icons.
   - Prose stays under 80ch.
   - Errors say what happened and how to fix it, and never apologise.
9. **Motion.** Only the What-if morph (450 ms ease-out), via `twin_morph.py`. Honour `prefers-reduced-motion`. No ambient animation.
10. **Depth.** Hairline borders. One shadow in the whole app: `.st-key-lever-tray`.
11. **Charts.**
    - Use `charts.forecast_figure` for any glucose curve, and pass `theme=None` to `st.plotly_chart`.
    - Never set explicit colours on chart text; CSS paints it from the live theme.
12. **Gates before every commit:** `pytest -q` (includes `tests/test_theme.py`, which runs the contrast audit and the config checks), `ruff check .`, `black --check .`.
