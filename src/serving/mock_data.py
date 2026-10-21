"""Seeded MOCK cohort: profiles, 5-min histories, meals and a seed food table.

Everything here is labelled MOCK (the genetic risk score is SYNTHETIC). It exists so
the dashboard, the alert engine and the what-if lab can be built and tested before
Person 3's processed data (Day 5) and meal engine (Day 10) land. It exposes the same
signatures the real providers must offer (see docs/contract_requests.md).

Observed glucose is generated from the same physiology the mock predictor uses
(src.serving.mock_predict), plus AR(1) sensor noise and, for P-011, an unmodelled
downward drift so the predictor has something real to anchor on.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import Meal, PatientProfile, PatientState, Provenance
from src.serving import timeseries as ts
from src.serving.mock_predict import Physio, model_curve

SEED = 20261120
IST = ts.IST
DAY_START = datetime(2026, 11, 18, 20, 0, tzinfo=IST)  # history starts the evening before
DAY_END = datetime(2026, 11, 19, 23, 55, tzinfo=IST)
REPLAY_START = datetime(2026, 11, 19, 6, 0, tzinfo=IST)
DEMO_NOW = datetime(2026, 11, 19, 20, 0, tzinfo=IST)
DEMO_PATIENT = "P-010"  # T2D, rice-heavy dinner at 20:00, short sleep
EARLIEST_NOW = DAY_START + timedelta(hours=4)
COLD_START_DAYS = 3

# --------------------------------------------------------------------------- food table

_SEED_SOURCE = "MOCK seed table (approximate, IFCT-inspired values; not authoritative)"
_FOOD_CSV = """food,portion_unit,grams_per_unit,carbs_g,protein_g,fat_g,fibre_g,gi,gl_per_unit,assumption_flag
white rice,katori,150,42,4,0.4,0.6,73,30.7,GI uses a polished-rice multiplier
brown rice,katori,150,36,4,1.3,2.7,55,19.8,
wheat roti,piece,40,18,3.5,0.8,2.6,62,11.2,
millet roti (bajra),piece,40,20,4,2,3.6,54,10.8,
jowar roti,piece,40,21,3.8,0.8,3.2,56,11.8,
ragi mudde,ball,100,26,2.5,0.5,3.5,67,17.4,
idli,piece,40,15,2,0.2,0.6,69,10.4,
dosa (plain),piece,80,29,4,4,1,66,19.1,
poha,katori,150,37,4,5,2,64,23.7,
upma,katori,150,33,5,7,2.5,62,20.5,
oats porridge,katori,150,20,4,3,3,55,11,
khichdi,katori,150,30,6,3,3,55,16.5,
veg biryani,plate,250,60,8,14,3,70,42,
sambar,katori,150,12,4.5,3,3.5,40,4.8,
dal (toor),katori,150,18,7,3,4,30,5.4,
rajma,katori,150,22,8,4,7,29,6.4,
chana masala,katori,150,25,8,6,7,28,7,
mixed veg sabzi,katori,150,10,2.5,6,4,35,3.5,
palak,katori,150,6,3,6,3,30,1.8,
paneer curry,katori,150,8,14,18,1.5,30,2.4,
chicken curry,katori,150,6,22,14,1,20,1.2,
boiled egg,piece,50,0.6,6,5,0,0,0,
curd,katori,100,4.5,3.5,4,0,36,1.6,
buttermilk,glass,200,4,2.5,1.5,0,30,1.2,
sprouts salad,katori,100,12,7,1,5,25,3,
banana,piece,100,23,1.1,0.3,2.6,51,11.7,
apple,piece,150,21,0.4,0.3,3.6,38,8,
masala chai (sugar),cup,150,12,2,2.5,0,60,7.2,
masala chai (no sugar),cup,150,4,2,2.5,0,30,1.2,
samosa,piece,60,24,3.5,10,2,60,14.4,
gulab jamun,piece,40,22,2,6,0.2,76,16.7,
papad,piece,15,9,3.5,0.5,1.5,60,5.4,
"""


@lru_cache(maxsize=1)
def food_db() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_FOOD_CSV), keep_default_na=False)
    df["source"] = _SEED_SOURCE
    df["licence"] = "MOCK (project-internal placeholder)"
    df["assumption_flag"] = df["assumption_flag"].astype(str)
    return df[
        [
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
        ]
    ]


def resolve_meal(items: list[dict[str, Any]], t: datetime | None = None, label: str | None = None) -> Meal:
    """Same signature as P3's ``resolve_meal(items)``; ``t``/``label`` are mock conveniences.

    items: [{"food": "wheat roti", "portion": 2, "unit": "piece"}]. ``unit`` may be the
    row's portion_unit or "g". Unknown foods/units raise ValueError.
    """
    if not items:
        raise ValueError("a meal needs at least one item")
    db = food_db().set_index("food")
    tot = dict(carbs_g=0.0, protein_g=0.0, fat_g=0.0, fibre_g=0.0, gl=0.0)
    flags, names = [], []
    for it in items:
        food, portion = it["food"], float(it.get("portion", 1))
        if food not in db.index:
            raise ValueError(f"unknown food {food!r}")
        if portion <= 0:
            raise ValueError(f"portion for {food!r} must be positive")
        row = db.loc[food]
        unit = it.get("unit") or row["portion_unit"]
        if unit == "g":
            n_units = portion / float(row["grams_per_unit"])
        elif unit == row["portion_unit"]:
            n_units = portion
        else:
            raise ValueError(f"unit {unit!r} not valid for {food!r} (use {row['portion_unit']!r} or 'g')")
        for k in ("carbs_g", "protein_g", "fat_g", "fibre_g"):
            tot[k] += n_units * float(row[k])
        tot["gl"] += n_units * float(row["gl_per_unit"])
        if row["assumption_flag"]:
            flags.append(f"{food}: {row['assumption_flag']}")
        names.append(f"{portion:g} {unit} {food}")
    return Meal(
        t=t or datetime(2026, 1, 1, tzinfo=UTC),
        label=label or " + ".join(names),
        items=[
            {
                "food": i["food"],
                "portion": float(i.get("portion", 1)),
                "unit": i.get("unit") or str(db.loc[i["food"], "portion_unit"]),
            }
            for i in items
        ],
        assumption_flags=flags or None,
        provenance=Provenance.MOCK,
        **{k: round(v, 2) for k, v in tot.items()},
    )


# --------------------------------------------------------------------------- meal templates

MEALS: dict[str, list[tuple[str, float, str]]] = {
    "idli": [("idli", 3, "piece"), ("sambar", 1, "katori")],
    "poha": [("poha", 1, "katori"), ("masala chai (sugar)", 1, "cup")],
    "dosa": [("dosa (plain)", 2, "piece"), ("sambar", 1, "katori")],
    "oats": [("oats porridge", 1, "katori"), ("masala chai (no sugar)", 1, "cup")],
    "upma": [("upma", 1, "katori"), ("masala chai (sugar)", 1, "cup")],
    "roti_dal": [
        ("wheat roti", 3, "piece"),
        ("dal (toor)", 1, "katori"),
        ("mixed veg sabzi", 1, "katori"),
        ("curd", 1, "katori"),
    ],
    "rice_sambar": [("white rice", 1.5, "katori"), ("sambar", 1, "katori"), ("curd", 1, "katori")],
    "rajma_rice": [("white rice", 1, "katori"), ("rajma", 1, "katori")],
    "biryani": [("veg biryani", 1, "plate"), ("curd", 1, "katori")],
    "chai_samosa": [("masala chai (sugar)", 1, "cup"), ("samosa", 1, "piece")],
    "fruit": [("apple", 1, "piece")],
    "sprouts": [("sprouts salad", 1, "katori"), ("buttermilk", 1, "glass")],
    "rice_heavy": [("white rice", 2, "katori"), ("sambar", 1, "katori"), ("curd", 1, "katori")],
    "roti_paneer": [("wheat roti", 2, "piece"), ("paneer curry", 1, "katori"), ("mixed veg sabzi", 1, "katori")],
    "khichdi": [("khichdi", 1.5, "katori"), ("curd", 1, "katori")],
    "chicken_rice": [("white rice", 1.5, "katori"), ("chicken curry", 1, "katori")],
    "millet": [("millet roti (bajra)", 2, "piece"), ("dal (toor)", 1, "katori"), ("palak", 1, "katori")],
}
MEAL_LABELS = {"b": "Breakfast", "l": "Lunch", "s": "Snack", "d": "Dinner"}


@dataclass(frozen=True)
class Spec:
    pid: str
    cohort: str
    sex: str
    age: int
    bmi: float
    hba1c: float
    fpg: float
    sleep_h: float
    usual_sleep: float
    plan: tuple[tuple[str, str, str], ...]  # (kind b/l/s/d, "HH:MM", template)
    walks: tuple[tuple[str, int], ...] = ()  # ("HH:MM", minutes)
    meds: tuple[str, ...] = ()
    comorbid: tuple[str, ...] = ()
    history_days: float = 12
    lipids: dict[str, float] = field(default_factory=dict)


def _p(*xs):
    return tuple(xs)


SPECS: tuple[Spec, ...] = (
    Spec(
        "P-001",
        "healthy",
        "F",
        29,
        21.5,
        5.1,
        92,
        7.5,
        7.4,
        _p(("b", "08:00", "idli"), ("l", "13:00", "roti_dal"), ("s", "16:30", "fruit"), ("d", "20:15", "roti_paneer")),
        walks=(("07:00", 30),),
        lipids={"ldl": 96, "hdl": 58, "tg": 92},
    ),
    Spec(
        "P-002",
        "healthy",
        "M",
        34,
        23.8,
        5.3,
        95,
        6.8,
        7.0,
        _p(
            ("b", "08:45", "poha"),
            ("l", "13:30", "rice_sambar"),
            ("s", "17:00", "chai_samosa"),
            ("d", "20:30", "khichdi"),
        ),
        lipids={"ldl": 112, "hdl": 46, "tg": 130},
    ),
    Spec(
        "P-003",
        "healthy",
        "F",
        41,
        22.9,
        5.4,
        92,
        7.2,
        7.2,
        _p(
            ("b", "08:15", "dosa"),
            ("l", "13:15", "rajma_rice"),
            ("s", "17:30", "sprouts"),
            ("d", "19:45", "roti_paneer"),
        ),
        walks=(("18:30", 25),),
        lipids={"ldl": 104, "hdl": 55, "tg": 101},
    ),
    Spec(
        "P-004",
        "healthy",
        "M",
        25,
        24.6,
        5.2,
        94,
        6.0,
        6.5,
        _p(("b", "09:00", "upma"), ("l", "13:45", "biryani"), ("s", "17:00", "fruit"), ("d", "21:00", "chicken_rice")),
        lipids={"ldl": 99, "hdl": 49, "tg": 118},
    ),
    Spec(
        "P-005",
        "prediabetes",
        "M",
        46,
        26.8,
        5.9,
        104,
        6.5,
        6.6,
        _p(
            ("b", "08:30", "poha"),
            ("l", "13:30", "rice_sambar"),
            ("s", "17:15", "chai_samosa"),
            ("d", "20:00", "chicken_rice"),
        ),
        comorbid=("dyslipidaemia",),
        lipids={"ldl": 138, "hdl": 41, "tg": 176},
    ),
    Spec(
        "P-006",
        "prediabetes",
        "F",
        51,
        27.5,
        6.1,
        108,
        7.0,
        7.0,
        _p(("b", "08:00", "idli"), ("l", "13:00", "roti_dal"), ("s", "16:45", "fruit"), ("d", "20:00", "khichdi")),
        walks=(("06:45", 30),),
        comorbid=("hypothyroidism",),
        lipids={"ldl": 126, "hdl": 50, "tg": 141},
    ),
    Spec(
        "P-007",
        "prediabetes",
        "M",
        39,
        28.9,
        6.0,
        110,
        6.2,
        6.4,
        _p(
            ("b", "08:30", "dosa"),
            ("l", "13:30", "biryani"),
            ("s", "17:00", "chai_samosa"),
            ("d", "19:30", "rice_heavy"),
        ),
        lipids={"ldl": 141, "hdl": 39, "tg": 192},
    ),
    Spec(
        "P-008",
        "prediabetes",
        "F",
        57,
        25.9,
        6.3,
        112,
        6.8,
        7.0,
        _p(
            ("b", "08:00", "oats"),
            ("l", "13:00", "rajma_rice"),
            ("s", "17:00", "sprouts"),
            ("d", "20:00", "roti_paneer"),
        ),
        walks=(("07:00", 20),),
        comorbid=("hypertension",),
        lipids={"ldl": 129, "hdl": 52, "tg": 150},
    ),
    Spec(
        "P-009",
        "t2d",
        "M",
        58,
        29.4,
        7.6,
        148,
        6.5,
        6.8,
        _p(
            ("b", "08:30", "upma"),
            ("l", "13:30", "rice_sambar"),
            ("s", "17:00", "chai_samosa"),
            ("d", "20:00", "chicken_rice"),
        ),
        meds=("metformin 500 mg BD",),
        comorbid=("hypertension", "dyslipidaemia"),
        lipids={"ldl": 132, "hdl": 38, "tg": 214},
    ),
    Spec(
        "P-010",
        "t2d",
        "F",
        54,
        30.2,
        8.4,
        146,
        5.5,
        7.1,
        _p(
            ("b", "08:30", "idli"),
            ("l", "13:30", "rice_sambar"),
            ("s", "17:00", "chai_samosa"),
            ("d", "20:00", "rice_heavy"),
        ),
        meds=("metformin 500 mg BD",),
        comorbid=("hypertension",),
        lipids={"ldl": 128, "hdl": 38, "tg": 210},
    ),
    Spec(
        "P-011",
        "t2d",
        "M",
        63,
        26.1,
        7.2,
        138,
        7.0,
        7.0,
        _p(("b", "08:00", "oats"), ("l", "13:00", "roti_dal"), ("s", "16:30", "fruit"), ("d", "20:30", "millet")),
        walks=(("18:00", 30),),
        meds=("metformin 1 g BD", "glimepiride 2 mg OD"),
        comorbid=("CKD stage 2",),
        lipids={"ldl": 110, "hdl": 44, "tg": 160},
    ),
    Spec(
        "P-012",
        "t2d",
        "F",
        49,
        32.5,
        9.1,
        168,
        6.0,
        6.3,
        _p(
            ("b", "09:00", "poha"),
            ("l", "14:00", "biryani"),
            ("s", "17:30", "chai_samosa"),
            ("d", "21:00", "rice_heavy"),
        ),
        meds=("metformin 1 g BD", "sitagliptin 100 mg OD"),
        comorbid=("hypertension", "obesity"),
        history_days=2,
        lipids={"ldl": 151, "hdl": 35, "tg": 248},
    ),
)
SPEC_BY_ID = {s.pid: s for s in SPECS}
USUAL = {s.pid: {"sleep_hours_prev": s.usual_sleep, "walk_minutes": 25.0 if s.walks else 5.0} for s in SPECS}

GAP = {"P-007": ("19:20", "20:05"), "P-004": ("11:10", "11:20")}  # CGM dropouts (local time, inclusive start)
HYPO_DRIFT_PID = "P-011"


def _at(hhmm: str, day: datetime = DEMO_NOW) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


def _grid() -> pd.DatetimeIndex:
    return pd.date_range(DAY_START, DAY_END, freq="5min")


def _hypo_drift(local: pd.DatetimeIndex) -> np.ndarray:
    """Unmodelled decline (e.g. sulfonylurea effect) from 16:30, recovering after 22:00 treatment."""
    x = (local - pd.Timestamp(_at("16:30"))).total_seconds().to_numpy() / 60
    d = -np.minimum(95.0, 7.0e-6 * np.clip(x, 0, None) ** 3)
    rec = (local - pd.Timestamp(_at("22:00"))).total_seconds().to_numpy() / 60
    d = np.where(rec > 0, d * np.clip(1 - rec / 90, 0, 1), d)
    return d


def _build(spec: Spec) -> tuple[PatientProfile, pd.DataFrame, list[Meal]]:
    rng = np.random.default_rng(SEED + int(spec.pid[2:]))
    grid = _grid()
    n = len(grid)
    tmin = (grid - grid[0]).total_seconds().to_numpy() / 60

    # sleep proxy and activity
    wake = _at("06:30") + timedelta(minutes=int(rng.integers(-20, 25)))
    onset = wake - timedelta(hours=spec.sleep_h)
    asleep = (grid >= pd.Timestamp(onset)) & (grid < pd.Timestamp(wake))
    mets = np.where(asleep, 0.95, 1.3) + rng.normal(0, 0.06, n)
    for hhmm, mins in spec.walks:
        s = pd.Timestamp(_at(hhmm))
        on = (grid >= s) & (grid < s + pd.Timedelta(minutes=mins))
        mets[on] = 3.5 + rng.normal(0, 0.15, on.sum())
    mets = np.clip(mets, 0.9, None)
    hr = 52 + 10 * mets + rng.normal(0, 2.5, n)

    meals = [
        resolve_meal(
            [{"food": f, "portion": p, "unit": u} for f, p, u in MEALS[tpl]],
            t=_at(hhmm).astimezone(UTC),
            label=f"{MEAL_LABELS[kind]}: {tpl.replace('_', ' ')}",
        )
        for kind, hhmm, tpl in spec.plan
    ]

    phys = Physio(fpg=spec.fpg, hba1c=spec.hba1c, bmi=spec.bmi, sleep_h=spec.sleep_h, t2d=spec.cohort == "t2d")
    meal_arr = np.array(
        [[(m.t - grid[0].to_pydatetime()).total_seconds() / 60, m.gl, m.fibre_g, m.fat_g] for m in meals]
    )
    g, _ = model_curve(phys, tmin, meal_arr, mets)

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 2.5)
    g = g + noise
    if spec.pid == HYPO_DRIFT_PID:
        g = g + _hypo_drift(grid)
    g = np.clip(g, 40, 400)

    missing = np.zeros(n, dtype=bool)
    if spec.pid in GAP:
        a, b = GAP[spec.pid]
        missing = (grid >= pd.Timestamp(_at(a))) & (grid < pd.Timestamp(_at(b)))
    g = np.where(missing, np.nan, np.round(g, 1))

    hist = pd.DataFrame(
        {
            "t": grid.tz_convert("UTC"),
            "glucose": g,
            "hr": np.round(hr, 1),
            "mets": np.round(mets, 2),
            "sleep_proxy": asleep.astype(float),
            "missing_mask": missing,
        }
    )

    k = int(spec.pid[2:])
    profile = PatientProfile(
        patient_id=spec.pid,
        cohort=spec.cohort,
        age=spec.age,
        sex=spec.sex,
        bmi=spec.bmi,
        hba1c=spec.hba1c,
        fasting_glucose=spec.fpg,
        lipids=spec.lipids,
        comorbidities=list(spec.comorbid),
        genetic_risk_score=round(float(np.clip(rng.beta(2, 3) + 0.15 * (spec.cohort == "t2d"), 0, 1)), 2),
        medications=list(spec.meds),
        pairing={
            "cgmacros_id": 100 + 3 * k,
            "match_distance": round(float(rng.uniform(0.15, 0.6)), 2),
            "match_on": ["status", "age", "bmi", "hba1c"],
            "note": "MOCK pairing",
        },
        history_days=spec.history_days,
        provenance={
            f: Provenance.MOCK
            for f in (
                "age",
                "sex",
                "bmi",
                "hba1c",
                "fasting_glucose",
                "lipids",
                "comorbidities",
                "medications",
                "pairing",
            )
        }
        | {"genetic_risk_score": Provenance.SYNTHETIC},
    )
    return profile, hist, meals


@lru_cache(maxsize=1)
def _cohort() -> dict[str, tuple[PatientProfile, pd.DataFrame, list[Meal]]]:
    return {s.pid: _build(s) for s in SPECS}


def list_patients() -> list[str]:
    return [s.pid for s in SPECS]


def clamp_now(now: datetime | None) -> datetime:
    now = (now or DEMO_NOW).astimezone(IST)
    now = min(max(now, EARLIEST_NOW), DAY_END)
    return ts.floor5(now)


def _uplink(g_now: float | None, stale: float) -> str:
    if stale > 30:
        return "OFFLINE"
    if g_now is not None and (g_now > 180 or g_now < 80):
        return "ALERT"
    return "ROUTINE"


def get_state(pid: str, now: datetime | None = None) -> PatientState:
    if pid not in SPEC_BY_ID:
        raise KeyError(f"unknown patient {pid!r}")
    now = clamp_now(now)
    profile, hist, meals = _cohort()[pid]
    h = hist[hist["t"] <= pd.Timestamp(now)].reset_index(drop=True)
    g_now = ts.glucose_now(h, now)
    return PatientState(
        profile=profile,
        history=h,
        meals=[m for m in meals if m.t <= now],
        now=now.astimezone(UTC),
        data_quality=ts.data_quality(h, now),
        stream_last_seen=ts.stream_last_seen(h, now),
        uplink_mode=_uplink(g_now, ts.stale_minutes(h, now)),
    )


def day_meals(pid: str) -> list[Meal]:
    """All meals of the replay day (for the meal timeline), including ones after ``now``."""
    return list(_cohort()[pid][2])


def replay_times(step_min: int = 15) -> list[datetime]:
    out, t = [], REPLAY_START
    while t <= DAY_END:
        out.append(t)
        t += timedelta(minutes=step_min)
    return out
