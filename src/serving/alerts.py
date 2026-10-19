"""Pure alert engine: tiers, confidence gating, episode dedupe, cooldown, escalation, budget.

Recall-first rules (architecture doc):
  * CRITICAL: current value < 70, OR q10 < 70 within 30 min, OR the rule net fired, OR the
    CGM stream is stale while the patient was recently high risk (status UNKNOWN).
  * WARNING: event probability ≥ stratum threshold AND confidence ≥ gate.
  * WATCH:   probability ≥ threshold but confidence < gate.
  * Confidence gating only demotes WARNING → WATCH; it never suppresses a hypo alert.
  * Stale data is never "low risk": a stale stream is at least WARNING (UNKNOWN).
  * Cooldown (default 60 min after an episode resolves) suppresses new WATCH episodes
    for the same patient and event. WARNING and CRITICAL always fire (ADR-010).
  * WATCH persisting for 2 consecutive evaluations escalates to WARNING.
  * Hysteresis: an episode resolves only after its condition is absent for 2 evaluations,
    so one rising episode stays one thread instead of flapping.

No Streamlit and no I/O, so the page only renders what this module returns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import IntEnum

from src.contracts import HYPO_THRESHOLD, Forecast, PatientState
from src.serving import timeseries as ts


class Tier(IntEnum):
    NONE = 0
    WATCH = 1
    WARNING = 2
    CRITICAL = 3


TIER_LABEL = {Tier.CRITICAL: "CRITICAL", Tier.WARNING: "WARNING", Tier.WATCH: "WATCH", Tier.NONE: "NO ALERT"}
STRATUM_BY_COHORT = {"healthy": "low", "prediabetes": "mid", "t2d": "high"}


@dataclass(frozen=True)
class AlertConfig:
    cooldown_min: int = 60
    stale_min: int = 20
    escalate_after: int = 2
    resolve_after: int = 2  # hysteresis: absent for N evaluations before an episode resolves
    budget_per_patient_day: float = 4.0  # 3–5 per patient-day total
    actionable_target: float = 2.0  # aim for < 2 WARNING/CRITICAL per patient-day
    recent_high_risk_min: int = 120


@dataclass(frozen=True)
class AlertInput:
    patient_id: str
    cohort: str
    now: datetime
    forecast: Forecast
    glucose_now: float | None
    data_quality: float | None
    stale_min: float

    @classmethod
    def from_state(cls, state: PatientState, fc: Forecast) -> AlertInput:
        return cls(
            patient_id=state.profile.patient_id, cohort=state.profile.cohort, now=state.now, forecast=fc,
            glucose_now=ts.glucose_now(state.history, state.now),
            data_quality=state.data_quality, stale_min=ts.stale_minutes(state.history, state.now),
        )


@dataclass(frozen=True)
class Candidate:
    patient_id: str
    cohort: str
    event: str  # "hypo" | "spike" | "unknown"
    tier: Tier
    probability: float | None
    confidence: float | None
    lead_min: int | None
    lead_label: str
    detectors: tuple[str, ...]
    data_quality: float | None
    model_version: str | None
    reason: str
    horizon_min: int = 120


@dataclass
class Alert:
    id: str
    patient_id: str
    cohort: str
    event: str
    tier: Tier
    probability: float | None
    confidence: float | None
    lead_min: int | None
    lead_label: str
    detectors: tuple[str, ...]
    data_quality: float | None
    model_version: str | None
    reason: str
    opened_at: datetime
    updated_at: datetime
    horizon_min: int = 120
    max_tier: Tier = Tier.NONE
    n_evals: int = 1
    escalated: bool = False
    resolved_at: datetime | None = None

    @property
    def status(self) -> str:
        if self.event == "unknown":
            return "UNKNOWN – needs attention"
        return "RESOLVED" if self.resolved_at else "ACTIVE"

    def age_min(self, now: datetime) -> int:
        return int((now - self.opened_at).total_seconds() // 60)


# --------------------------------------------------------------------------- single evaluation


def stratum(cohort: str) -> str:
    return STRATUM_BY_COHORT.get(cohort, "high")


def _lead(fc: Forecast, event: str, g_now: float | None) -> tuple[int | None, str]:
    """Earliest predicted crossing: q50 first, then the band edge; else a horizon bucket."""
    curve = ts.forecast_curve(fc, g_now)
    if event == "spike":
        t = ts.first_crossing(curve, "q50", 180, above=True) or ts.first_crossing(curve, "q90", 180, above=True)
    else:
        t = ts.first_crossing(curve, "q50", HYPO_THRESHOLD, above=False) or ts.first_crossing(
            curve, "q10", HYPO_THRESHOLD, above=False)
    if t is None:
        return None, "≤120 min"
    if fc.trajectory:
        return t, f"~{t} min"
    bucket = next(b for b in (30, 60, 120) if t <= b)
    return t, f"≤{bucket} min"


def evaluate(inp: AlertInput, thresholds: dict, cfg: AlertConfig = AlertConfig(),
             recently_high_risk: bool = False) -> list[Candidate]:
    fc = inp.forecast
    thr = thresholds["strata"][stratum(inp.cohort)]
    gate = float(thresholds.get("confidence_gate", 0.5))
    flags = fc.detector_flags or {}
    fired = tuple(k for k, v in flags.items() if v)
    out: list[Candidate] = []

    def cand(event: str, tier: Tier, prob: float | None, reason: str) -> Candidate:
        lead_min, lead_label = _lead(fc, event, inp.glucose_now) if event != "unknown" else (None, "now")
        return Candidate(inp.patient_id, inp.cohort, event, tier, prob, fc.confidence, lead_min, lead_label,
                         fired, inp.data_quality, fc.model_version, reason)

    # ---- hypo (recall-first: never suppressed by confidence)
    curve = ts.forecast_curve(fc, inp.glucose_now)
    q10_30 = curve.loc[curve["t_min"] <= 30, "q10"].min()
    rule_hypo = bool(flags.get("rule_net")) and (inp.glucose_now is None or inp.glucose_now <= 120)
    if inp.glucose_now is not None and inp.glucose_now < HYPO_THRESHOLD:
        out.append(cand("hypo", Tier.CRITICAL, fc.hypo_prob, f"current glucose {inp.glucose_now:.0f} mg/dL < 70"))
    elif q10_30 < HYPO_THRESHOLD:
        out.append(cand("hypo", Tier.CRITICAL, fc.hypo_prob, "lower band (q10) below 70 within 30 min"))
    elif rule_hypo:
        out.append(cand("hypo", Tier.CRITICAL, fc.hypo_prob, "rule net fired (low and falling)"))
    elif fc.hypo_prob >= thr["hypo"]:
        tier = Tier.WARNING if fc.confidence >= gate else Tier.WATCH
        note = "" if tier == Tier.WARNING else " (low confidence: demoted to WATCH, not suppressed)"
        out.append(cand("hypo", tier, fc.hypo_prob, f"hypo probability {fc.hypo_prob:.0%} ≥ {thr['hypo']:.0%}{note}"))

    # ---- spike
    rule_spike = bool(flags.get("rule_net")) and inp.glucose_now is not None and inp.glucose_now > 120
    if rule_spike:
        out.append(cand("spike", Tier.CRITICAL, fc.spike_prob, "rule net fired (very high glucose)"))
    elif fc.spike_prob >= thr["spike"]:
        tier = Tier.WARNING if fc.confidence >= gate else Tier.WATCH
        note = "" if tier == Tier.WARNING else " (low confidence: WATCH)"
        out.append(cand("spike", tier, fc.spike_prob, f"spike probability {fc.spike_prob:.0%} ≥ {thr['spike']:.0%}{note}"))

    # ---- stale stream → UNKNOWN, never low risk
    if inp.stale_min >= cfg.stale_min:
        high = recently_high_risk or stratum(inp.cohort) == "high" or any(
            c.tier >= Tier.WARNING for c in out)
        stale_txt = "no CGM data" if inp.stale_min == float("inf") else f"no CGM data for {inp.stale_min:.0f} min"
        out.append(cand("unknown", Tier.CRITICAL if high else Tier.WARNING, None,
                        f"{stale_txt}; risk cannot be assessed" + (" (recently high risk)" if high else "")))
    return out


# --------------------------------------------------------------------------- episodes


@dataclass
class AlertBook:
    """Stateful episode tracker. Deterministic: same inputs in the same order → same alerts."""

    cfg: AlertConfig = field(default_factory=AlertConfig)
    active: dict[tuple[str, str], Alert] = field(default_factory=dict)
    history: list[Alert] = field(default_factory=list)  # every episode ever opened
    cooldown_until: dict[tuple[str, str], datetime] = field(default_factory=dict)
    watch_streak: dict[tuple[str, str], int] = field(default_factory=dict)
    absent: dict[tuple[str, str], int] = field(default_factory=dict)
    last_high_risk: dict[str, datetime] = field(default_factory=dict)
    suppressed: int = 0

    def recently_high_risk(self, pid: str, now: datetime) -> bool:
        t = self.last_high_risk.get(pid)
        return t is not None and now - t <= timedelta(minutes=self.cfg.recent_high_risk_min)

    def update(self, now: datetime, candidates: Iterable[Candidate],
               evaluated: Iterable[str] | None = None) -> list[Alert]:
        """Apply one evaluation round. ``evaluated`` lists patients assessed this round; their
        active episodes with no candidate resolve. Defaults to the candidates' patients."""
        cands = {(c.patient_id, c.event): c for c in candidates}
        seen = set(evaluated) if evaluated is not None else {k[0] for k in cands}

        for key, alert in list(self.active.items()):
            if key[0] in seen and key not in cands:
                self.absent[key] = self.absent.get(key, 0) + 1
                if self.absent[key] < self.cfg.resolve_after:
                    continue
                self.absent.pop(key, None)
                alert.resolved_at = now
                self.cooldown_until[key] = now + timedelta(minutes=self.cfg.cooldown_min)
                self.watch_streak.pop(key, None)
                del self.active[key]

        for key, c in cands.items():
            self.absent.pop(key, None)
            if c.tier >= Tier.WARNING and c.event != "unknown":
                self.last_high_risk[c.patient_id] = now
            tier = c.tier
            streak = self.watch_streak.get(key, 0) + 1 if tier == Tier.WATCH else 0
            self.watch_streak[key] = streak
            escalated = tier == Tier.WATCH and streak >= self.cfg.escalate_after
            if escalated:
                tier = Tier.WARNING

            if key in self.active:
                a = self.active[key]
                a.tier, a.probability, a.confidence = tier, c.probability, c.confidence
                a.lead_min, a.lead_label, a.detectors = c.lead_min, c.lead_label, c.detectors
                a.data_quality, a.model_version, a.reason = c.data_quality, c.model_version, c.reason
                a.updated_at, a.n_evals = now, a.n_evals + 1
                a.max_tier = max(a.max_tier, tier)
                a.escalated = a.escalated or escalated
                continue

            in_cooldown = now < self.cooldown_until.get(key, now)
            if in_cooldown and tier < Tier.WARNING:
                self.suppressed += 1
                continue
            a = Alert(
                id=f"{c.patient_id}-{c.event}-{now.astimezone(ts.IST):%Y%m%d%H%M}", patient_id=c.patient_id,
                cohort=c.cohort, event=c.event, tier=tier, probability=c.probability, confidence=c.confidence,
                lead_min=c.lead_min, lead_label=c.lead_label, detectors=c.detectors, data_quality=c.data_quality,
                model_version=c.model_version, reason=c.reason, opened_at=now, updated_at=now,
                horizon_min=c.horizon_min, max_tier=tier, escalated=escalated,
            )
            self.active[key] = a
            self.history.append(a)
        return self.queue()

    def queue(self) -> list[Alert]:
        return sort_queue(self.active.values())


def sort_queue(alerts: Iterable[Alert]) -> list[Alert]:
    """Tier desc, then lead time asc (unknown lead last), then probability desc."""
    return sorted(alerts, key=lambda a: (-int(a.tier), a.lead_min if a.lead_min is not None else 10_000,
                                         -(a.probability or 0)))


# --------------------------------------------------------------------------- burden


@dataclass(frozen=True)
class Burden:
    episodes: int
    actionable: int
    critical: int
    patient_days: float
    per_patient_day: float
    actionable_per_patient_day: float
    critical_share: float
    within_budget: bool
    suppressed: int


def burden(book: AlertBook, n_patients: int, window_hours: float) -> Burden:
    patient_days = max(n_patients * window_hours / 24, 1e-9)
    eps = book.history
    actionable = sum(a.max_tier >= Tier.WARNING for a in eps)
    critical = sum(a.max_tier == Tier.CRITICAL for a in eps)
    per = len(eps) / patient_days
    return Burden(
        episodes=len(eps), actionable=actionable, critical=critical, patient_days=round(patient_days, 2),
        per_patient_day=round(per, 2), actionable_per_patient_day=round(actionable / patient_days, 2),
        critical_share=round(critical / len(eps), 3) if eps else 0.0,
        within_budget=per <= book.cfg.budget_per_patient_day, suppressed=book.suppressed,
    )


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class Snapshot:
    now: datetime
    queue: tuple[Alert, ...]
    burden: Burden


def replay(times: list[datetime], patients: list[str], get_state: Callable[[str, datetime], PatientState],
           predict: Callable[[PatientState], Forecast], thresholds: dict,
           cfg: AlertConfig = AlertConfig()) -> list[Snapshot]:
    """Scrub ``now`` through ``times``, evaluating every patient each step."""
    book = AlertBook(cfg=cfg)
    snaps: list[Snapshot] = []
    for now in times:
        cands: list[Candidate] = []
        for pid in patients:
            state = get_state(pid, now)
            inp = AlertInput.from_state(state, predict(state))
            cands += evaluate(inp, thresholds, cfg, recently_high_risk=book.recently_high_risk(pid, state.now))
        book.update(now, cands, evaluated=patients)
        hours = max((now - times[0]).total_seconds() / 3600, 0.25)
        snaps.append(Snapshot(now, tuple(replace(a) for a in book.queue()), burden(book, len(patients), hours)))
    return snaps
