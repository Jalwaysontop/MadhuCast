# Architecture decision records

Short ADRs: context, decision, consequence. Newest at the bottom.

## ADR-001: Repository name
- **Context:** The work split names the repo `MadhuTwin_TeamName/`. The summit brief requires `TeamName_CollegeName`.
- **Decision:** The brief wins at submission. All code uses paths relative to the repo root (`backend.ROOT` is derived from `__file__`), so renaming the folder or repo is safe.
- **Consequence:** Rename before making the repo public. The README notes the rename.

## ADR-002: FHIR library version (provisional)
- **Context:** `fhir.resources` 7.x+ defaults to R5. `fhir.resources.R4B` provides R4B. Strict R4 needs the 6.x line. It is not installed yet.
- **Decision:** Provisionally, pin `fhir.resources` 7.x and build with `fhir.resources.R4B`. R4B is wire-compatible with R4 for Patient, Observation, RiskAssessment and Bundle. The README will say "R4B" honestly rather than "R4". To be confirmed when F9 is built (Days 11–13). If a judge requires strict R4, fall back to 6.5.x.
- **Consequence:** One import path (`fhir.resources.R4B`) across `fhir_export.py` and its tests.

## ADR-003: Split backend selection (data vs model)
- **Context:** Person 3's data (Day 5/10) and Person 2's model (Day 14) arrive separately.
- **Decision:** `MADHUTWIN_BACKEND` sets the default. `MADHUTWIN_DATA` and `MADHUTWIN_MODEL` override it per subsystem. Env vars are read on every call (no import-time capture), so tests can monkeypatch them.
- **Consequence:** Mixed modes are possible, e.g. real profiles with the mock model. The sidebar shows both.

## ADR-004: Caching lives in the dashboard, not the backend
- **Context:** The prompt suggests `st.cache_resource` / `st.cache_data` in the backend.
- **Decision:** `backend.py` stays Streamlit-free and uses `functools.lru_cache` for files and model import. Pages wrap backend calls in `st.cache_data` / `st.cache_resource`.
- **Consequence:** The backend is unit-testable without a Streamlit runtime. `backend.clear_caches()` exists for tests and for hot-swapping data.

## ADR-005: Scenario levers are applied by P1, not inside the model
- **Context:** `predict(state, scenario)` could apply levers either inside the model or before it.
- **Decision:** `src/serving/scenario.py` (P1, pure) rewrites `PatientState` (meals, METs, sleep proxy), and the model sees an ordinary state. The scenario is passed through for logging only.
- **Consequence:** The what-if logic is identical for mock and real models, and testable in isolation. P2's model needs no what-if-specific code. Raised as R2.1 so P2 can object.

## ADR-006: Additive contract fields are Optional; validation is strict on shape
- **Context:** The work split defines `predict()` loosely. The UI needs more (trajectory, versions, detector flags).
- **Decision:** All additions are Optional. Models use `extra="forbid"`, so typos and unrequested fields (e.g. a medication lever) fail loudly. Quantile ordering and range are validated, and violations raise `ContractError`, which renders as an error card.
- **Consequence:** P2 and P3 get immediate, specific errors from `pytest -m contract` instead of silent UI breakage. Adding a new field requires a contract version bump.

## ADR-007: New file locations for time series and meals
- **Context:** The split specifies profile JSONs but not where the 5-min series and meal lists live.
- **Decision:** Proposed `data/processed/timeseries/{pid}.parquet` and `data/processed/meals/{pid}.json` (R3.2, R3.3).
- **Consequence:** Pending P3 confirmation. If P3 picks other paths, only `backend._real_state` changes.

## ADR-002 (update): FHIR version confirmed
- **Decision:** `fhir.resources==8.1.0`, building with `fhir.resources.R4B` (FHIR 4.3.0). Every bundle is parsed back before download, and a test round-trips it. The README says "R4B".
- **Consequence:** If strict R4 is demanded, pin `fhir.resources` 6.5.x and switch the imports. The resource shapes used are unchanged.

## ADR-008: Forecasts as RiskAssessment + preliminary Observation
- **Context:** Modelling a prediction as a plain Observation is a semantic stretch.
- **Decision:** Spike and hypo probabilities go in a `RiskAssessment` (one `prediction` per outcome, with `probabilityDecimal` and `whenPeriod`). The quantile curve goes in an `Observation` with status `preliminary`, method "MadhuTwin model estimate (version)", and one component per quantile and horizon. CGM readings use LOINC 99504-3 (verified against the HL7 CGM IG profile), UCUM mg/dL, category `laboratory`. The Patient carries only the pseudonym and is tagged `HTEST` plus `synthetic`.
- **Consequence:** Consumers can tell measured values from model estimates. Bundle ids are uuid5-derived, so the committed sample is reproducible.

## ADR-009: Extra P1-owned modules
- **Context:** The layout lists a fixed set of P1 files. A few more pure modules kept pages thin and testable.
- **Decision:** Added `src/serving/timeseries.py` (freshness, data quality, sleep window, curve interpolation and crossings, shared by mock, real adapter, alerts and pages), `src/serving/logbook.py` (feedback/audit JSONL), and `dashboard/components/{common,charts,health,tables}.py`. All are P1-owned. No P2/P3 file was touched.
- **Consequence:** P2 and P3 can reuse `timeseries.forecast_curve` and the crossing helpers if they want identical semantics.

## ADR-010: Cooldown suppresses WATCH re-alerts only; hysteresis on resolution
- **Context:** In replay, a 60-min cooldown after a snack-spike episode suppressed the demo patient's dinner-spike WARNING. Under a recall-first design that is the wrong trade.
- **Decision:**
  - An episode resolves only after its condition is absent for 2 consecutive evaluations (hysteresis), which kills flapping.
  - Cooldown (default 60 min, configurable) suppresses only new WATCH episodes for the same patient and event. WARNING and CRITICAL always open.
  - Suppressed candidates are counted and shown.
- **Consequence:** On the mock day, burden is 2.6 alerts per patient-day against a budget of 4. Actionable alerts are 2.6 against a target under 2, which the UI shows honestly as "needs tuning".

## ADR-011: Lever semantics, planned activity, and evaluation at meal time
- **Context:** A post-meal walk or a later meal lies in the future relative to `now`, so the history alone can't carry it. An earlier meal can't be "observed" after the fact.
- **Decision:**
  - Added the optional contract field `PatientState.planned_activity: list[ActivityBlock]`. Meals may have `t > now` (planned).
  - The what-if lab evaluates at the selected meal's time, so the 2-hour window covers that meal.
  - A negative time shift moves the evaluation time back to the new meal time (history truncated, at least 4 h kept).
  - The walk lever adds a 3.5-MET block starting `offset` minutes after the meal, and raises METs/HR for any past part of it.
  - The sleep lever rewrites only the previous-night window of the sleep proxy and the METs/HR it derives from.
- **Consequence:** Raised to P2 in `contract_requests.md` (R2.3). If the real model ignores planned activity, the walk sanity test in `whatif_sanity.csv` will show it.

## ADR-012: SciPy pinned to 1.16.2
- **Context:** The SciPy 1.15.3 macOS wheel fails to `dlopen` on the dev machine (zero-fill `__thread_bss` section error).
- **Decision:** Pin `scipy==1.16.2`.
- **Consequence:** P2 should use the same pin or newer.

## ADR-013: Mock attributions go through the same pipeline as what-if
- **Context:** The first mock attributions held the CGM anchor fixed, but a what-if re-anchors to observed glucose. The agreement check on the trust page caught it: 71% on the mock (sleep, and meals hours in the past, were overstated).
- **Decision:** Model-input features are perturbed through the full predict pipeline, re-anchoring included. Only the two anchor features (`glucose_now`, `glucose_slope_30m`) are perturbed directly.
- **Consequence:** Mock agreement is now 100% (11/11 at demo time), and a test pins it. This is exactly the check the real model must pass. Below 100% on the real model should be investigated, not hidden.

## ADR-014: Deterministic demo mode
- **Decision:** `MADHUTWIN_DEMO=1` pins P-010 and replay time 20:00 IST. The mock cohort is seeded (`20261120`), and predictions are deterministic (tested).
- **Consequence:** The video can be re-recorded identically, and screenshots in `docs/figures/` match it.

## Cut-line check (Day 12): to be recorded with the team
- Status on P1 side: F4, F6, F7, F9 and F10 are all built on mock. Nothing from P1's cut list has been cut. Record the team decision here on Day 12.
