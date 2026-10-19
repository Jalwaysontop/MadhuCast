"""Pure helpers over PatientState history and Forecast curves (no Streamlit, no I/O).

Shared by the mock backend, the real-data adapter, the alert engine and the pages,
so every consumer computes freshness, data quality and crossings the same way.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.contracts import HYPO_THRESHOLD, SPIKE_THRESHOLD, Forecast

IST = ZoneInfo("Asia/Kolkata")
STEP_MIN = 5
FRESH_MIN = 15  # a CGM sample older than this is not "current"
QUALITY_WINDOW_MIN = 240


def floor5(ts: datetime) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % STEP_MIN, second=0, microsecond=0)


def hhmm(ts: datetime) -> str:
    return ts.astimezone(IST).strftime("%H:%M")


def _t64(history: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(history["t"], utc=True).to_numpy(dtype="datetime64[ns]")


def _now64(now: datetime) -> np.datetime64:
    return pd.Timestamp(now).tz_convert("UTC").tz_localize(None).to_datetime64()


def _upto(history: pd.DataFrame, now: datetime) -> pd.DataFrame:
    return history[_t64(history) <= _now64(now)]


def last_glucose(history: pd.DataFrame, now: datetime) -> tuple[datetime, float] | None:
    t = _t64(history)
    g = history["glucose"].to_numpy(dtype=float)
    ok = (t <= _now64(now)) & ~history["missing_mask"].to_numpy(dtype=bool) & ~np.isnan(g)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return None
    i = idx[-1]
    return pd.Timestamp(t[i]).tz_localize("UTC").to_pydatetime(), float(g[i])


def glucose_now(history: pd.DataFrame, now: datetime, fresh_min: int = FRESH_MIN) -> float | None:
    """Latest CGM value if it is fresh; None means UNKNOWN (never 'low risk')."""
    last = last_glucose(history, now)
    if last is None or (now - last[0]) > timedelta(minutes=fresh_min):
        return None
    return last[1]


def stale_minutes(history: pd.DataFrame, now: datetime) -> float:
    last = last_glucose(history, now)
    if last is None:
        return float("inf")
    return max(0.0, (now - last[0]).total_seconds() / 60)


def data_quality(history: pd.DataFrame, now: datetime, window_min: int = QUALITY_WINDOW_MIN) -> float:
    """Fraction of expected 5-min CGM samples present in the trailing window."""
    t = pd.to_datetime(history["t"])
    w = history[(t > pd.Timestamp(now) - pd.Timedelta(minutes=window_min)) & (t <= pd.Timestamp(now))]
    expected = window_min / STEP_MIN
    present = int((~w["missing_mask"].astype(bool) & w["glucose"].notna()).sum())
    return float(min(1.0, present / expected))


def stream_last_seen(history: pd.DataFrame, now: datetime) -> dict[str, datetime]:
    h = _upto(history, now)
    out: dict[str, datetime] = {}
    cgm = last_glucose(history, now)
    if cgm:
        out["cgm"] = cgm[0]
    wear = h[h["hr"].notna()]
    if not wear.empty:
        out["wearable"] = pd.Timestamp(wear["t"].iloc[-1]).to_pydatetime()
    return out


def glucose_slope(history: pd.DataFrame, now: datetime, window_min: int = 30) -> float | None:
    """Least-squares slope of observed glucose (mg/dL per min) over the trailing window."""
    t = pd.to_datetime(history["t"])
    w = history[(t > pd.Timestamp(now) - pd.Timedelta(minutes=window_min)) & (t <= pd.Timestamp(now))]
    w = w[~w["missing_mask"].astype(bool) & w["glucose"].notna()]
    if len(w) < 3:
        return None
    x = (pd.to_datetime(w["t"]) - pd.Timestamp(now)).dt.total_seconds().to_numpy() / 60
    return float(np.polyfit(x, w["glucose"].to_numpy(dtype=float), 1)[0])


def night_window(now: datetime) -> tuple[datetime, datetime]:
    """The 'previous night' window: 20:00 the evening before to 10:00 (local), or to now if earlier."""
    local = now.astimezone(IST)
    day10 = local.replace(hour=10, minute=0, second=0, microsecond=0)
    return day10 - timedelta(hours=14), min(local, day10)


def prior_sleep_hours(history: pd.DataFrame, now: datetime) -> float | None:
    """Hours flagged by the sleep *proxy* in the previous-night window."""
    start, end = night_window(now)
    t = pd.to_datetime(history["t"])
    w = history[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))]
    if w.empty:
        return None
    return float((w["sleep_proxy"].fillna(0) >= 0.5).sum() * STEP_MIN / 60)


def missing_runs(history: pd.DataFrame) -> list[tuple[datetime, datetime]]:
    """Contiguous runs of missing CGM samples as (first_missing, last_missing + 5 min)."""
    m = history["missing_mask"].astype(bool).to_numpy()
    t = pd.to_datetime(history["t"]).to_list()
    runs, start = [], None
    for i, flag in enumerate(m):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(m) - 1):
            end = i if not flag else i + 1
            runs.append((t[start].to_pydatetime(), (t[end - 1] + pd.Timedelta(minutes=STEP_MIN)).to_pydatetime()))
            start = None
    return runs


# --------------------------------------------------------------------------- forecast curves

CURVE_COLUMNS = ("t_min", "q05", "q10", "q50", "q90", "q95")


def forecast_curve(fc: Forecast, glucose_at_now: float | None = None) -> pd.DataFrame:
    """5-min forecast curve. Uses ``trajectory`` when present, else a monotone (PCHIP)
    interpolation through now → 30 → 60 → 120 and sets ``attrs['interpolated']``.
    """
    if fc.trajectory:
        df = pd.DataFrame([p.model_dump() for p in fc.trajectory])
        interpolated = False
    else:
        knots_t = [30, 60, 120]
        qs = {name: [fc.horizon(h)[i] for h in knots_t] for i, name in enumerate(("q10", "q50", "q90"))}
        if fc.quantiles_90:
            qs["q05"] = [fc.quantiles_90[str(h)][0] for h in knots_t]
            qs["q95"] = [fc.quantiles_90[str(h)][1] for h in knots_t]
        if glucose_at_now is not None:
            knots_t = [0, *knots_t]
            qs = {k: [glucose_at_now, *v] for k, v in qs.items()}
        grid = np.arange(STEP_MIN, 121, STEP_MIN)
        df = pd.DataFrame({"t_min": grid})
        for k, v in qs.items():
            df[k] = PchipInterpolator(knots_t, v, extrapolate=True)(grid)
        interpolated = True
    for col in ("q05", "q95"):
        if col not in df:
            df[col] = np.nan
        df[col] = df[col].astype(float)
    # enforce ordering after interpolation
    df["q10"] = np.minimum(df["q10"], df["q50"])
    df["q90"] = np.maximum(df["q90"], df["q50"])
    df["q05"] = np.where(df["q05"].notna(), np.minimum(df["q05"], df["q10"]), np.nan)
    df["q95"] = np.where(df["q95"].notna(), np.maximum(df["q95"], df["q90"]), np.nan)
    df = df[list(CURVE_COLUMNS)].sort_values("t_min").reset_index(drop=True)
    df.attrs["interpolated"] = interpolated
    return df


def first_crossing(curve: pd.DataFrame, col: str, threshold: float, above: bool) -> int | None:
    hit = curve[curve[col] > threshold] if above else curve[curve[col] < threshold]
    return int(hit["t_min"].iloc[0]) if not hit.empty else None


def spike_crossing(curve: pd.DataFrame) -> int | None:
    return first_crossing(curve, "q50", SPIKE_THRESHOLD, above=True)


def hypo_crossing(curve: pd.DataFrame) -> int | None:
    return first_crossing(curve, "q50", HYPO_THRESHOLD, above=False)


def time_above(curve: pd.DataFrame, threshold: float = SPIKE_THRESHOLD, col: str = "q50") -> int:
    return int((curve[col] > threshold).sum() * STEP_MIN)
