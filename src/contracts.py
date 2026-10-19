"""Cross-team interface contracts for MadhuTwin.

Every page reads data only through these models (via ``src.serving.backend``).
Units: glucose mg/dL, durations in minutes, timestamps tz-aware (stored UTC,
displayed Asia/Kolkata).

Fields marked "requested addition" are Optional: the original work split did
not define them, so the UI must degrade gracefully when they are absent.
See docs/contract_requests.md.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "0.1.1"

GLUCOSE_MIN, GLUCOSE_MAX = 20.0, 600.0  # physiological sanity bounds for rejection
SPIKE_THRESHOLD, HYPO_THRESHOLD = 180.0, 70.0
HORIZONS_MIN = (30, 60, 120)
HISTORY_COLUMNS = ("t", "glucose", "hr", "mets", "sleep_proxy", "missing_mask")
MIN_HISTORY_MIN = 240


class Provenance(str, Enum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    MOCK = "MOCK"
    MODEL_ESTIMATE = "MODEL_ESTIMATE"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


def _require_tz(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return v


# --------------------------------------------------------------------------- P3 → P1


class PatientProfile(_Strict):
    """Static stream. P3 writes data/processed/patient_profiles/{patient_id}.json."""

    patient_id: str = Field(pattern=r"^P-\d{3}$")  # pseudonym only
    cohort: Literal["healthy", "prediabetes", "t2d"]
    age: int = Field(ge=18, le=100)
    sex: Literal["M", "F"]
    bmi: float = Field(ge=12, le=70)
    hba1c: float = Field(ge=3, le=18)  # %
    fasting_glucose: float = Field(ge=GLUCOSE_MIN, le=GLUCOSE_MAX)
    lipids: dict[str, float] | None = None  # mg/dL, e.g. {"ldl": 120, "hdl": 42, "tg": 180}
    comorbidities: list[str] = []
    genetic_risk_score: float | None = None  # always SYNTHETIC
    medications: list[str] = []  # display only; never a what-if lever
    pairing: dict[str, Any] | None = None  # {"cgmacros_id": ..., "match_distance": ...}
    history_days: float | None = None  # requested addition: for cold-start "limited history"
    provenance: dict[str, Provenance]

    @model_validator(mode="after")
    def _grs_is_synthetic(self) -> PatientProfile:
        if self.genetic_risk_score is not None and self.provenance.get("genetic_risk_score") != Provenance.SYNTHETIC:
            raise ValueError("genetic_risk_score must carry SYNTHETIC provenance")
        return self


class Meal(_Strict):
    t: datetime
    label: str
    carbs_g: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    fibre_g: float = Field(ge=0)
    gl: float = Field(ge=0)
    items: list[dict[str, Any]] | None = None  # [{"food": "roti", "portion": 2, "unit": "piece"}]
    assumption_flags: list[str] | None = None  # requested addition: surfaces ⚠ rows from food DB
    provenance: Provenance

    _tz = field_validator("t")(_require_tz)


class ActivityBlock(_Strict):
    """A planned or counterfactual activity bout (e.g. a post-meal walk)."""

    start: datetime
    minutes: int = Field(ge=1, le=180)
    mets: float = Field(default=3.5, ge=1, le=12)
    kind: str = "walk"

    _tz = field_validator("start")(_require_tz)


class PatientState(_Strict):
    """The ``patient`` argument to ``predict()``.

    ``history`` is a pandas DataFrame on a 5-min grid covering at least 4 h, with
    columns HISTORY_COLUMNS. ``missing_mask`` is True where the glucose sample
    is absent (glucose may then be NaN). ``meals`` may include planned meals
    with ``t > now`` (what-if meal-time shifts).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    profile: PatientProfile
    history: pd.DataFrame
    meals: list[Meal]
    now: datetime
    data_quality: float | None = Field(default=None, ge=0, le=1)
    stream_last_seen: dict[str, datetime] | None = None
    uplink_mode: str | None = None  # requested addition: e.g. "ALERT" / "ROUTINE" from P3's edge sim
    planned_activity: list[ActivityBlock] = []  # requested addition: future bouts (what-if walks)

    _tz = field_validator("now")(_require_tz)

    @field_validator("history")
    @classmethod
    def _check_history(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"history missing columns: {missing}")
        if len(df) < 2:
            raise ValueError("history needs at least 2 rows")
        t = pd.to_datetime(df["t"])
        if t.dt.tz is None:
            raise ValueError("history.t must be timezone-aware")
        if not t.is_monotonic_increasing:
            raise ValueError("history.t must be sorted ascending")
        span = (t.iloc[-1] - t.iloc[0]).total_seconds() / 60
        if span < MIN_HISTORY_MIN - 5:
            raise ValueError(f"history spans {span:.0f} min; need >= {MIN_HISTORY_MIN}")
        return df


# --------------------------------------------------------------------------- P1 → P2


class Scenario(_Strict):
    """What-if levers. ``Scenario()`` (all defaults) is the baseline.

    Medication fields are deliberately absent (safety); ``extra="forbid"``
    rejects any attempt to pass one.
    """

    meal_swap: dict[str, Any] | None = None  # {"meal_index": i, "items": [...]}
    portion_scale: float = Field(default=1.0, ge=0.25, le=2.0)
    walk_minutes: int = Field(default=0, ge=0, le=60)
    walk_start_offset_min: int = Field(default=15, ge=0, le=120)
    meal_time_shift_min: int = Field(default=0, ge=-120, le=120)
    prior_sleep_hours: float | None = Field(default=None, ge=3, le=10)
    target_meal_index: int | None = None  # requested addition: lever target; None = latest meal

    def is_baseline(self) -> bool:
        return self == Scenario()


# --------------------------------------------------------------------------- P2 → P1

Quantiles3 = tuple[float, float, float]


class TrajectoryPoint(_Strict):
    t_min: int = Field(ge=0, le=240)
    q05: float | None = None
    q10: float
    q50: float
    q90: float
    q95: float | None = None


class Forecast(_Strict):
    """Return value of ``predict()``. Glucose tuples are (q10, q50, q90)."""

    glucose_30: Quantiles3
    glucose_60: Quantiles3
    glucose_120: Quantiles3
    spike_prob: float = Field(ge=0, le=1)
    hypo_prob: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    shap_values: dict[str, float] = {}
    attention_weights: list[float] | list[list[float]] = []
    # ---- requested additions (Optional; see docs/contract_requests.md) ----
    trajectory: list[TrajectoryPoint] | None = None
    quantiles_90: dict[str, tuple[float, float]] | None = None  # {"30": (q05, q95), ...}
    feature_values: dict[str, float] | None = None
    feature_baselines: dict[str, float] | None = None
    detector_flags: dict[str, bool] | None = None  # {"quantile":..,"classifier":..,"rule_net":..}
    model_version: str | None = None
    feature_lib_version: str | None = None
    provenance: Provenance = Provenance.MODEL_ESTIMATE

    @field_validator("glucose_30", "glucose_60", "glucose_120")
    @classmethod
    def _ordered(cls, q: Quantiles3) -> Quantiles3:
        if not all(GLUCOSE_MIN <= x <= GLUCOSE_MAX for x in q):
            raise ValueError(f"glucose quantile out of range: {q}")
        if not q[0] <= q[1] <= q[2]:
            raise ValueError(f"quantiles must satisfy q10 <= q50 <= q90: {q}")
        return q

    @field_validator("trajectory")
    @classmethod
    def _traj_ordered(cls, pts: list[TrajectoryPoint] | None) -> list[TrajectoryPoint] | None:
        if pts is None:
            return pts
        for p in pts:
            lo = p.q05 if p.q05 is not None else p.q10
            hi = p.q95 if p.q95 is not None else p.q90
            if not lo <= p.q10 <= p.q50 <= p.q90 <= hi:
                raise ValueError(f"trajectory quantiles out of order at t_min={p.t_min}")
        if [p.t_min for p in pts] != sorted(p.t_min for p in pts):
            raise ValueError("trajectory must be sorted by t_min")
        return pts

    def horizon(self, minutes: int) -> Quantiles3:
        return {30: self.glucose_30, 60: self.glucose_60, 120: self.glucose_120}[minutes]


# --------------------------------------------------------------------------- P2 reports/

REPORT_FILES: dict[str, str] = {
    "eval_summary": "eval_summary.json",
    "ablation": "ablation.csv",
    "calibration": "calibration.csv",
    "subgroups": "subgroups.csv",
    "clarke_grid": "clarke_grid.csv",
    "thresholds": "thresholds.json",
    "whatif_sanity": "whatif_sanity.csv",
}

REPORT_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "ablation": ("variant", "metric", "value", "ci_low", "ci_high", "n_participants"),
    "calibration": ("kind", "nominal", "empirical", "n"),
    "subgroups": ("cohort", "metric", "value", "ci_low", "ci_high", "n_participants"),
    "clarke_grid": ("zone", "share"),
    "whatif_sanity": ("test", "passed", "n_cases", "pass_rate"),
}

FOOD_DB_COLUMNS = (
    "food",
    "portion_unit",
    "grams_per_unit",
    "carbs_g",
    "protein_g",
    "fat_g",
    "fibre_g",
    "gi",
    "gl_per_unit",
    "source",
    "licence",
    "assumption_flag",
)


class ContractError(ValueError):
    """Raised when a cross-team payload fails validation. Pages render it as an error card."""

    def __init__(self, source: str, detail: str):
        self.source = source
        self.detail = detail
        super().__init__(f"[{source}] {detail}")
