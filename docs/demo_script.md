# MadhuTwin demo video: script and shot list (≥ 20 min)

**Setup before recording**
```bash
MADHUTWIN_DEMO=1 MADHUTWIN_BACKEND=mock streamlit run dashboard/app.py   # or MADHUTWIN_BACKEND=real after Day 14
```
- Browser at 1440×900, light or dark theme (both are tested). Zoom 100%. Close other tabs.
- Demo mode pins patient **P-010** (T2D, rice-heavy dinner at 20:00, 5.5 h sleep) and replay time **20:00 IST**.
- `logs/` empty before recording (`rm logs/*.jsonl`) so the audit trail shown on screen is fresh.
- If running on mock, say "mock" out loud once in the demo segment. The badges show it anyway.

**Speaker roles:** P1 drives the dashboard. P2 presents the model and ablation. P3 presents data and pairing.

| Time | Segment | Speaker | Visual |
|---|---|---|---|
| 0:00–1:30 | Title, team, college, one-line positioning | All | Title slide |
| 1:30–4:00 | Problem: post-meal spikes, rice-heavy dinners, hypo risk on sulfonylureas, clinicians blind between HbA1c visits | P1 | Slides |
| 4:00–7:00 | Data: CGMacros (REAL), Synthea EHR (SYNTHETIC), nearest-neighbour pairing, Indian meal augmentation, sleep proxy, licences (CGMacros NonCommercial question) | P3 | Slides + Patient card pairing disclosure |
| 7:00–10:00 | Architecture: 7 layers, contracts, mock→real switch, privacy boundary, Pull stub | P1 | `docs/architecture_diagram.pdf`, `docs/privacy_boundary.pdf` |
| 10:00–14:00 | Model: FiLM conditioning, physics prior, quantile heads, conformal calibration; ablation with CIs | P2 | Slides + `docs/figures/` |
| **14:00–18:00** | **Live demo** (below) | P1 | Dashboard |
| 18:00–20:30 | Model trust: calibration, subgroups, Clarke grid, what-if sanity, agreement rate, honest limits | P2 + P1 | Model Trust page |
| 20:30–22:00 | Competitors and what we don't claim; future scope (federated, ABDM); close | All | Slides |

---

## Live demo (14:00–18:00): exact clicks and lines

### 1. Alert queue (14:00–14:50)
- **Click:** nothing. The app opens on *Alert Queue*.
- **Say:** "This is the triage view at 8 pm. The engine has replayed the whole day every 15 minutes. Top of the queue: **P-011, CRITICAL hypo**. The lower edge of the forecast band drops below 70 within 30 minutes. Second: **P-007 is UNKNOWN**. Its CGM has been silent for 45 minutes. We never show missing data as low risk."
- **Point at:** the header metrics. "Alert burden is **2.6 per patient-day against a budget of 4**. Actionable alerts are 2.6 against a target under 2, so these placeholder thresholds, badged **UNCALIBRATED**, still need tuning. That's the honest number."
- **Point at:** P-007's WATCH spike row. "Low confidence demotes an alert to WATCH. It never hides a hypo."
- **Drag** the sidebar *Replay time* slider from 20:00 to 13:30, then back to 20:00. "The queue evolves live."

### 2. Drill into the patient (14:50–15:20)
- **Click:** expand **P-010 · spike · WARNING**, then **Open patient**.
- **Say:** "P-010: 54-year-old woman, T2D, HbA1c 8.4. Medications are display only. They are never a what-if lever."
- **Click:** sidebar *Patient* page (optional, 10 s). Point at the **SYNTHETIC** genetic risk score and the pairing disclosure.

### 3. Forecast band (15:20–16:00)
- **Click:** *Forecast*.
- **Say:** "Blue is the last 4 hours of CGM. Orange is the 2-hour forecast with 80% and 90% bands. She just ate 2 katori of white rice with sambar and curd. The model expects her to cross 180 at about 8:10 pm and peak around 270."
- **Point at:** the risk badges and the active alert. **Click** **Useful** on the alert (feedback is logged with the model version).

### 4. Rice → millet roti + walk (16:00–17:00)
- **Click:** *What-if Lab*. Point at the yellow disclaimer: "Model estimate from observational data, not causal, not medical advice."
- **Click:** preset **Rice → millet roti**. "Equal carbohydrate, more fibre: the peak drops by about 45 mg/dL."
- **Click:** preset **Combined** (swap + 20-min walk 15 min after the meal).
- **Read** the delta strip: "**−61 mg/dL peak, 30 fewer minutes above 180.** The bands don't overlap, so the difference is more than noise *for this model*."
- **Toggle** *Overlay on one chart*.
- **Click:** **Short sleep (5 h)**. "Short sleep pushes the peak up slightly. Notice the 'bands overlap' warning: we don't overclaim small effects."
- **Click:** **Save scenario for comparison** after Combined. "Clinicians can keep up to four side by side."

### 5. Why panel (17:00–17:30)
- **Click:** *Why*.
- **Say:** "The meal's glycaemic load dominates, then HbA1c and BMI, and last night's 5.5 hours of sleep versus her usual 7.1 adds a little more. These sentences come from templates, not a language model."
- **Point at:** the reliability table. "We re-run the matching what-ifs to check that each attribution points the same way the simulator moves. ✓ across the board."

### 6. FHIR export (17:30–17:45)
- **Click:** *Forecast*, scroll to **FHIR export**, expand *Show bundle*.
- **Say:** "A FHIR R4B bundle: pseudonymous Patient tagged as test data, CGM Observations with LOINC 99504-3, a RiskAssessment for spike and hypo, and the forecast as a preliminary Observation. Validated before download."
- **Click:** **Request raw window** (with consent ticked). "A stub. It writes an audit line, and no raw data leaves the device."

### 7. Trust page (17:45–18:00, then continues in 18:00–20:30)
- **Click:** *Model Trust*.
- **Say:** "Every number here carries a provenance badge. The ablation highlights a winner only if its confidence intervals don't overlap. Right now none is highlighted, and we say that plainly."

---

## Shot list (B-roll / screenshots)
1. Alert queue, full page (`docs/figures/ui/alert_queue_light_1440.png`)
2. Replay slider scrubbing 13:30 → 20:00 (screen recording)
3. Forecast page with bands and meal markers
4. What-if Combined, side-by-side (`docs/figures/ui/whatif_light_1440.png`)
5. What-if overlay toggle
6. Why panel: attribution bars and reliability table
7. FHIR bundle expander
8. Model Trust: ablation table and limits list
9. Architecture and privacy diagrams (PDF, full screen)
10. P-007 forecast showing UNKNOWN current glucose and the hatched CGM gap

## Fallbacks during recording
- If a page shows a red **Contract error** card on the real backend, switch the failing subsystem back to mock (`MADHUTWIN_MODEL=mock`), restart, and say so on camera.
- If the cold start is slow, open the Alert Queue once before recording. The replay is cached afterwards.
