# Contract requests (Person 1 → Person 2, Person 3)

Source of truth: `src/contracts.py` (contract version **0.1.1**). This page explains it in plain terms.
Everything marked *requested addition* is **Optional**. The dashboard works without it, just with less detail.
Reply in this file or in chat. Person 1 will not edit your files.

Conventions for everyone:
- Glucose in **mg/dL**, durations in **minutes**.
- Timestamps are **timezone-aware**, stored in UTC. The UI converts them to Asia/Kolkata for display.
- Patient IDs are pseudonyms matching `P-\d{3}` (e.g. `P-017`). No names, no addresses.
- Every displayed number carries a provenance label: `REAL` / `SYNTHETIC` / `MOCK` / `MODEL_ESTIMATE`.

---

## To Person 3 (data). Day 5: processed data. Day 10: meal engine.

### R3.1 Patient profiles: `data/processed/patient_profiles/{patient_id}.json`
One JSON per patient, matching `PatientProfile`:

```json
{
  "patient_id": "P-017", "cohort": "t2d", "age": 52, "sex": "M",
  "bmi": 28.4, "hba1c": 8.1, "fasting_glucose": 162,
  "lipids": {"ldl": 128, "hdl": 38, "tg": 210},
  "comorbidities": ["hypertension"],
  "genetic_risk_score": 0.71,
  "medications": ["metformin 500 mg BD"],
  "pairing": {"cgmacros_id": 23, "match_distance": 0.42, "match_on": ["status","age","bmi","hba1c"]},
  "history_days": 10,
  "provenance": {"age": "SYNTHETIC", "hba1c": "SYNTHETIC", "genetic_risk_score": "SYNTHETIC"}
}
```
- `genetic_risk_score`, if present, **must** have `"provenance": {"genetic_risk_score": "SYNTHETIC"}`. The validator rejects it otherwise.
- *Requested addition:* `history_days` (days of CGM available). The UI uses it to show a "limited history" (cold start) badge.

### R3.2 Time series: `data/processed/timeseries/{patient_id}.parquet` (*new file; the split didn't specify a location*)
- A 5-min grid with columns `t, glucose, hr, mets, sleep_proxy, missing_mask`.
- `t` is tz-aware UTC.
- `missing_mask` is bool, True where glucose is absent. Keep the row and set glucose to NaN. **Do not interpolate gaps.** The UI draws them as breaks.
- `sleep_proxy` is derived from night METs and HR. The UI labels it a proxy.
- At least 4 h of history is needed before any `now` we query.

### R3.3 Meals: `data/processed/meals/{patient_id}.json` (*new file*)
- A list of `Meal` objects: `t, label, carbs_g, protein_g, fat_g, fibre_g, gl, items, provenance`.
- `provenance` is `REAL` for CGMacros meals and `SYNTHETIC` for the Indian augmentation.

### R3.4 Food DB: `data/indian_food_db.csv`
- Columns: `food, portion_unit, grams_per_unit, carbs_g, protein_g, fat_g, fibre_g, gi, gl_per_unit, source, licence, assumption_flag`.
- Macros are **per portion_unit**.
- `assumption_flag` is empty, or a short reason such as `"rice polishing multiplier"`. The UI shows ⚠ on non-empty values.
- Please fill `licence` per row (IFCT/INDB licence is still to be verified).

### R3.5 Meal engine: `src/data/meal_engine.py`
```python
def resolve_meal(items: list[dict]) -> Meal | dict
# items: [{"food": "roti", "portion": 2, "unit": "piece"}, {"food": "dal", "portion": 1, "unit": "katori"}]
```
- Unknown food or unit: raise `ValueError` with a clear message. The UI shows it.
- *Requested addition:* `Meal.assumption_flags: list[str]`, the flags from any rows used.
- Returning a plain dict is fine; the backend validates it.
- Until Day 10, `src/serving/mock_data.py` provides the same signature on top of a 30-item seed table.

### R3.6 (optional) `PatientState.uplink_mode`
If the edge simulation produces an uplink mode (`ROUTINE` / `ALERT` / `OFFLINE`), expose it per patient. The UI only displays it.

---

## To Person 2 (model). Day 14: `src/serving/predict_api.py`

### R2.1 `predict(state: PatientState, scenario: Scenario) -> Forecast | dict`
- `state` is a `PatientState` (see `src/contracts.py`). `state.history` is a pandas DataFrame.
- `scenario` is a `Scenario`. `Scenario()` (all defaults) means baseline.
- **Person 1 applies the levers** before calling you (`src/serving/scenario.py` rewrites meals, the activity series and the sleep proxy). So you can ignore `scenario`, or use it only for logging. If you'd rather apply levers inside the model, say so now.
- **Medication is intentionally absent** from `Scenario`. Don't add it.
- Load model weights once at import or behind your own cache. The backend imports the module once.

Required fields (from the work split): `glucose_30/60/120` as `(q10, q50, q90)`, `spike_prob`, `hypo_prob`, `confidence`, `shap_values`, `attention_weights`.
Validation the backend enforces:
- q10 ≤ q50 ≤ q90,
- glucose in [20, 600],
- probabilities and confidence in [0, 1].

Requested additions (Optional, in priority order):
1. `model_version`, `feature_lib_version`: shown on every alert and in the FHIR export.
2. `trajectory`: a list of `{t_min, q10, q50, q90, q05?, q95?}` every 5 min out to 120. Without it the UI interpolates the 3 horizons and labels the curve "interpolated".
3. `detector_flags`: `{"quantile": bool, "classifier": bool, "rule_net": bool}`. These drive the alert tiers; `rule_net=True` forces Critical.
4. `feature_values`, `feature_baselines`: the raw value behind each SHAP key, plus the patient's typical value. They enable sentences like "slept 5.2 h vs your usual 7.1 h".
5. `quantiles_90`: `{"30": [q05, q95], "60": [...], "120": [...]}`, for the 90% band.

**SHAP keys:** please use the names in the feature registry, which Person 1 will publish in `src/explain/templates.py` (`FEATURE_REGISTRY`). Proposed keys: `meal_gl`, `meal_carbs`, `meal_fibre`, `meal_fat`, `minutes_since_meal`, `walk_minutes`, `mets_1h`, `hr_mean_1h`, `sleep_hours_prev`, `hba1c`, `bmi`, `age`, `glucose_now`, `glucose_slope_30m`. Unknown keys still render, with the raw key as the label.

### R2.2 `reports/` (written by `scripts/evaluate.py`; P1 reads only)
| File | Shape |
|---|---|
| `eval_summary.json` | `{"horizons": {"30": {"rmse": [v, lo, hi], "mae": [...]}, ...}, "event": {"spike": {"recall":..,"pr_auc":..,"f2":..}, "hypo": {...}}, "lead_time_min": {"median":..,"p25":..}, "false_alarms_per_patient_day":.., "coverage": {"80":.., "90":..}, "recall_at_budget":..}` |
| `ablation.csv` | `variant, metric, value, ci_low, ci_high, n_participants` (variants: `ehr_only, wearable_only, concat, film, film_physics`; also `glucose_only` if you have it) |
| `calibration.csv` | `kind, nominal, empirical, n`, where `kind` ∈ `coverage`, `reliability_spike`, `reliability_hypo` |
| `subgroups.csv` | `cohort, metric, value, ci_low, ci_high, n_participants` |
| `clarke_grid.csv` | `zone, share` (zones A–E, shares sum to 1) |
| `thresholds.json` | `{"strata": {"low": {"spike": p, "hypo": p}, "mid": {...}, "high": {...}}, "confidence_gate": c, "stratum_rule": "<text>"}` |
| `whatif_sanity.csv` | `test, passed, n_cases, pass_rate` |

If a file is missing, the trust page shows "not yet available". On mock, it shows mock values labelled MOCK.

Notes on `whatif_sanity.csv`:
- `passed` is a **count** of passing cases, not a bool.
- Expected tests (the mock already computes these live, so they double as a reference):
  - `more_carbs_raises_peak`
  - `walk_lowers_peak`
  - `short_sleep_raises_peak`
  - `millet_swap_lowers_peak`

### R2.3 Planned activity and planned meals (contract 0.1.1, additive)
- `PatientState.planned_activity: list[ActivityBlock]` holds future bouts, each with `start`, `minutes`, `mets` (3.5 for a brisk walk) and `kind`. This is how the post-meal-walk lever reaches the model.
- `PatientState.meals` may include meals with `t > now` (a later meal time is a *planned* meal).
- If your model can't use either, say so. The walk/meal-time levers then fall back to "no effect" on the real model, and the sanity test will show it.

### R2.4 Attribution semantics
- Please attribute the **predicted 2-hour peak** (max of q50 over the horizon) or say what output you attribute, so the template wording stays correct.
- The trust page runs an attribution ↔ what-if sign-agreement check. Attributions computed against a fixed anchor tend to disagree with what-if re-anchoring (we hit this on the mock; see ADR-013).

---

## Integration gate
`pytest -m contract` runs the same assertions against `MADHUTWIN_BACKEND=real`. Run it as soon as you push. Failures get logged below.

## Open items / failure log
| Date | From | Item | Status |
|---|---|---|---|
| 2026-10-19 | P1 | Contract v0.1.0 circulated | awaiting review |
| 2026-10-19 | P1 | v0.1.1 (additive): `ActivityBlock`, `PatientState.planned_activity`, planned meals (R2.3) | awaiting review |
| 2026-10-19 | P1 | `pytest -m contract` against real: **fails** (no `data/processed/patient_profiles/`). Expected until the Day-5 P3 handoff | open |
