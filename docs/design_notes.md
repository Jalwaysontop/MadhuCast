# Design notes: "Jamun"

Speaker notes for the two-minute "design choices" moment in the video, and the record of what we tried.

**Figma source of truth:** [MadhuTwin — Jamun UI](https://www.figma.com/design/lGkCcYYcV1PHiNWpD1pOjG). It has three pages:
- *Foundations*: variables `Jamun / light`, `Jamun / dark` and `Jamun / size`, plus 11 text styles.
- *Components*: Lucide icon set, TierBadge, ProvBadge, Button, PresetSegment, NavItem, VitalReadout.
- *Screens*: Alert queue, Forecast (light and dark), What-if lab, Why.

Code tokens live in `dashboard/theme/tokens.py`.

## The idea
In Indian households, **jamun** is a fruit traditionally linked with managing blood sugar. Its deep purple gives the product a colour that belongs to this subject and no other.

Everything else is a calm clinical instrument:
- cool porcelain surfaces and ink-dark numerals,
- generous whitespace,
- colour reserved for meaning only: jamun for the twin, teal for the 70–180 target range, and a strict severity scale.

## The one bold element: the Twin Curve
Glucose history (slate) flows into a translucent jamun forecast cone (80% band inside the 90% band) over the shaded 70–180 mg/dL target band.
- Thresholds are dashed, with right-edge labels.
- "Now" is a thin vertical line.
- A ring marks the predicted crossing ("Spike ~10 min").
- Meals sit on the time axis, sized by glycaemic load.
- Missing CGM is a hatched UNKNOWN region, never an interpolated line.

All the visual ambition lives here; everything around it is quiet.

## The one orchestrated motion
In the What-if lab, changing a lever **morphs** the dashed magenta scenario curve from its previous shape to the new one (450 ms, ease-out) while the "Δ peak" label counts to its new value.
- `st.plotly_chart` remounts on every rerun, which kills transitions. So `dashboard/components/twin_morph.py` renders a small Plotly.js iframe (jsDelivr `plotly.js-dist-min@2.35.2`).
- The iframe draws the previous scenario, kept in `st.session_state`, and `Plotly.animate`s to the new one.
- With `prefers-reduced-motion: reduce` the swap is instant.
- There is no other ambient animation.

## Tokens (light / dark)
| Token | Light | Dark | Role |
|---|---|---|---|
| porcelain | #F4F6F7 | #0F121A | app background |
| surface | #FFFFFF | #151925 | panels |
| ink / ink-muted | #1B2130 / #5A6275 | #E8EAF0 / #A3AAB8 | text, numerals |
| jamun | #5B2A6E | #B48ACB | brand, twin, primary |
| target | #1F7268 | #4FB3A6 | 70–180 range |
| critical / warning / watch | #B3261E / #8F5100 / #3A6EA5 | #FF8A80 / #F0B45A / #8AB4E8 | severity |
| unknown | #5F6673, dashed/hatched | #9CA3AF | stale / missing |
| scenario | #C2185B, dashed | #F48FB1 | what-if curve |

- **Type:** Hanken Grotesk 400–700 with tabular numerals everywhere, on a 1.25 scale (12 / 14 / 16 / 20 / 25 / 31 / 39). Noto Sans Devanagari is used only for Hindi food names (चावल white rice, बाजरा रोटी millet roti).
- **Radii:** 2 px for chips and badges, 8 px for panels, 999 px only for the header risk pill.
- **Depth:** hairline borders. The one shadow in the app is the What-if lever tray.

`python scripts/check_contrast.py` audits every text/background pair in both themes at WCAG AA. Tinted badges are blended over their surface before measuring.

## What we tried and rejected
- **Server-side theme detection for charts.** `st.context.theme` is documented as unreliable on first load and after a menu change. Charts built for dark stayed dark after switching to light: near-invisible history line, dark gridlines. A teammate called this out. Now:
  - data marks use a **theme-proof mid-tone chart palette** (each colour ≥ 3:1 on both themes' surfaces, enforced by the audit),
  - chart text, grid and hover are painted by CSS from live `--mt-*` variables,
  - a zero-height script sets `data-mt-theme` from the app's real background, so a theme switch restyles everything without a rerun.
- **Coloured chart labels.** At 12 px, coloured labels can't hit 4.5:1 on both themes. Labels are now ink, and colour stays on the lines (dashed thresholds, ring markers). Severity is never colour alone anyway.
- **Original token values.** `#A15C00` (warning), `#6B7280` (unknown) and `#2F8F83` (target) failed AA on their own tints in light mode. They were darkened to the values above.
- **Two charts side by side as the default What-if view.** Overlay compares faster. Split view remains a toggle.
- **A Google Fonts URL with two families in Streamlit's `theme.font`.** Streamlit rejects it, and the session fails silently into a legacy navigation. A test (`tests/test_theme.py`) now guards it.
- **Emoji icons, ALL-CAPS labels, `A · B · C` meta strings, gradient washes, and a hover lift on every card.** Removed in favour of Lucide icons at a 1.75 stroke, sentence case and plain sentences.

## Screens that may still read as "Streamlit" (escape hatch)
- **Widget chrome:** the multiselect chips, slider handles and data editor follow Streamlit's components, themed but not redrawn. That's acceptable for a PoC.
- **Sidebar navigation:** styled with CSS on `data-testid` hooks rather than built by us.
- **Post-submission proposal:** a React + FastAPI front end. `src/serving/backend.py` already isolates the contract, so the migration would touch no model or data code. We are not starting it before the summit.
