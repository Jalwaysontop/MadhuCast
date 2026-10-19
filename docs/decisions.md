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
