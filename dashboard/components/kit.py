"""Jamun component kit. One function per component; pages compose these and hold no raw
HTML/CSS. Mirrors the Figma components (TierBadge, ProvBadge, VitalReadout, …).

Every readout carries `data-label` / `data-value` attributes so tests and screen
readers get the number without parsing styling.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from dashboard.theme.tokens import FOOD_HINDI, ICONS, PROV_COLOR, TIER_COLOR, TIER_ICON, Mode

esc = html.escape


def mode() -> Mode:
    """Active Streamlit theme ('light' | 'dark'); falls back to light outside a session."""
    try:
        t = st.context.theme.type
        return "dark" if t == "dark" else "light"
    except Exception:
        return "light"


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def icon(name: str, size: int = 16, label: str | None = None) -> str:
    aria = f'role="img" aria-label="{esc(label)}"' if label else 'aria-hidden="true"'
    return (
        f'<svg class="mt-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" {aria}>{ICONS[name]}</svg>'
    )


# --------------------------------------------------------------------------- badges


def provenance_badge(kind: str) -> str:
    """REAL solid · SYNTHETIC flask · MOCK dashed · MODEL_ESTIMATE jamun. Min 12px, AA contrast."""
    k = kind.value if hasattr(kind, "value") else str(kind)
    label = {"MODEL_ESTIMATE": "Model estimate", "REAL": "Real", "SYNTHETIC": "Synthetic", "MOCK": "Mock"}.get(
        k, k.title()
    )
    color = PROV_COLOR.get(k, "ink_muted").replace("_", "-")
    ic = icon("flask-conical", 12) if k == "SYNTHETIC" else ""
    cls = "mt-badge mock" if k == "MOCK" else "mt-badge"
    return f'<span class="{cls}" style="color:var(--mt-{color})" data-prov="{k}">{ic}{esc(label)}</span>'


def tier_badge(tier: str) -> str:
    """CRITICAL / WARNING / WATCH / UNKNOWN / CLEAR: icon + word + colour, never colour alone."""
    t = tier.upper()
    color = TIER_COLOR.get(t, "unknown")
    word = {"CLEAR": "No alert"}.get(t, t.capitalize())
    cls = "mt-tier unknown" if t == "UNKNOWN" else "mt-tier"
    return (
        f'<span class="{cls}" style="color:var(--mt-{color});background:var(--mt-{color}-tint)" data-tier="{t}">'
        f"{icon(TIER_ICON.get(t, 'info'), 14)}{esc(word)}</span>"
    )


def pill(text: str, tier: str) -> str:
    t = tier.upper()
    color = TIER_COLOR.get(t, "unknown")
    return (
        f'<span class="mt-pill" style="color:var(--mt-{color});background:var(--mt-{color}-tint)">'
        f"{icon(TIER_ICON.get(t, 'info'), 14)}{esc(text)}</span>"
    )


def deva(food: str) -> str:
    """'रोटी roti' style label: Devanagari name (if known) beside the English one."""
    hi = FOOD_HINDI.get(food)
    return f'<span class="mt-deva" lang="hi">{esc(hi)}</span> {esc(food)}' if hi else esc(food)


# --------------------------------------------------------------------------- header / footer


def header_bar(
    pid: str,
    cohort: str,
    risk: tuple[str, str],
    dq: float | None,
    uplink: str | None,
    backend: str,
    now: datetime | str,
) -> None:
    risk_text, risk_tier = risk
    bars = 0 if dq is None else sum(dq >= x for x in (0.25, 0.5, 0.75, 0.9))
    meter_cls = "mt-meter" if (dq or 0) >= 0.9 else ("mt-meter warn" if (dq or 0) >= 0.6 else "mt-meter bad")
    meter = (
        f'<span class="{meter_cls}" aria-hidden="true">'
        + "".join(f'<i class="{"on" if i < bars else ""}" style="height:{6 + 3 * i}px"></i>' for i in range(4))
        + "</span>"
    )
    dq_txt = "unknown" if dq is None else f"{dq:.0%}"
    now_txt = now if isinstance(now, str) else now.strftime("%H:%M")
    md(
        f'<div class="mt-header" role="banner">'
        f'<span><span class="mt-pid">{esc(pid)}</span> &nbsp;{esc(cohort)}</span>'
        f'<span data-risk="{esc(risk_text)}">{pill(risk_text, risk_tier)}</span>'
        f"<span>Data quality {meter} <b>{dq_txt}</b></span>"
        f"<span>Uplink <b>{esc((uplink or 'not provided').capitalize())}</b></span>"
        f'<span>Backend <b>{esc(backend)}</b></span><span class="mt-spacer"></span>'
        f'<span>{icon("clock", 14)} Now <b>{esc(now_txt)} IST</b></span></div>'
    )


def footer() -> None:
    legend = " ".join(provenance_badge(k) for k in ("REAL", "SYNTHETIC", "MOCK", "MODEL_ESTIMATE"))
    md(
        f'<div class="mt-footer"><span>Provenance</span> {legend}'
        "<b>Synthetic/open data only. Research PoC, not a medical device.</b>"
        "<span>Model estimates are learned from observational data and are not causal effects.</span></div>"
    )


# --------------------------------------------------------------------------- readouts


@dataclass(frozen=True)
class Readout:
    label: str
    value: str
    unit: str = ""
    foot: str = ""
    trend: str | None = None  # "up" | "down"
    tone: str | None = None  # "good" | "bad" (colours the numeral)
    badge: str = ""  # pre-rendered badge HTML for the label row
    foot_tone: str | None = None  # "warn"


def readout_html(r: Readout, extra_cls: str = "") -> str:
    trend = ""
    if r.trend:
        color = "warning" if r.trend == "up" else "target"
        trend = f'<span style="color:var(--mt-{color})">{icon("trending-up" if r.trend == "up" else "trending-down", 20, f"trend {r.trend}")}</span>'
    tone = f" {r.tone}" if r.tone else ""
    foot_cls = f"mt-r-foot {r.foot_tone}" if r.foot_tone else "mt-r-foot"
    return (
        f'<div class="mt-readout {extra_cls}" data-label="{esc(r.label)}" data-value="{esc(r.value)}">'
        f'<div class="mt-r-label">{esc(r.label)} {r.badge}</div>'
        f'<div class="mt-r-row"><span class="mt-r-value{tone}">{esc(r.value)}</span>'
        f'<span class="mt-r-unit">{esc(r.unit)}</span>{trend}</div>'
        f'<div class="{foot_cls}">{esc(r.foot)}</div></div>'
    )


def readouts(items: Sequence[Readout], alert_html: str | None = None) -> None:
    md('<div class="mt-readouts">' + "".join(readout_html(r) for r in items) + (alert_html or "") + "</div>")


def alert_card(tier: str, headline: str, detail: str) -> str:
    crit = " critical" if tier.upper() == "CRITICAL" else ""
    return (
        f'<div class="mt-readout alert{crit}" data-label="Active alert" data-value="{esc(headline)}">'
        f'{tier_badge(tier)}<div style="font-weight:600;font-size:16px;margin:8px 0 2px">{esc(headline)}</div>'
        f'<div class="mt-r-foot">{esc(detail)}</div></div>'
    )


def delta_strip(items: Sequence[Readout]) -> None:
    md(
        '<div class="mt-strip" role="group" aria-label="Scenario deltas">'
        + "".join(readout_html(r) for r in items)
        + "</div>"
    )


# --------------------------------------------------------------------------- prose / states


def disclaimer(text: str) -> None:
    md(
        f'<div class="mt-disclaimer" role="note"><span style="color:var(--mt-jamun)">{icon("triangle-alert", 16)}</span>'
        f'<span class="mt-prose">{esc(text)}</span></div>'
    )


def empty_state(what: str, how_to_fix: str) -> None:
    md(f'<div class="mt-empty" role="status"><b>{esc(what)}</b><span class="mt-muted">{esc(how_to_fix)}</span></div>')


def lead(text: str) -> None:
    md(f'<p class="mt-lead">{esc(text)}</p>')


def caption(text: str) -> None:
    md(f'<div class="mt-caption mt-prose">{esc(text)}</div>')


def title_row(title: str, badges: Iterable[str] = (), aside: str = "") -> None:
    b = " ".join(badges)
    a = f'<span class="mt-muted" style="margin-left:auto">{esc(aside)}</span>' if aside else ""
    md(
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:4px 0 8px">'
        f'<h1 style="margin:0;padding:0;font-size:31px;font-weight:700;letter-spacing:-0.015em">{esc(title)}</h1>{b}{a}</div>'
    )


def definition_list(rows: Sequence[tuple[str, str, str]]) -> str:
    """rows: (label, value, badge_html)."""
    items = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)} {b}</dd></div>" for k, v, b in rows)
    return f'<dl class="mt-dl">{items}</dl>'


def bar(frac: float, color: str) -> str:
    frac = max(0.0, min(1.0, frac))
    return f'<span class="mt-bar" aria-hidden="true"><i style="width:{frac * 100:.0f}%;background:var(--mt-{color})"></i></span>'


# --------------------------------------------------------------------------- driver bars (Why panel)


def driver_bars(groups: Sequence[tuple[str, Sequence[tuple[str, str, float]]]], scale_max: float) -> str:
    """Diverging bars: raises → right in warning, lowers ← left in target. groups: (group, [(label, value, effect)])."""
    rows = []
    for g, items in groups:
        rows.append(f'<div class="mt-group">{esc(g)}</div>')
        for label, value, eff in items:
            w = min(1.0, abs(eff) / max(scale_max, 1e-9)) * 50
            color = "warning" if eff > 0 else "target"
            side = f"left:50%;width:{w:.1f}%" if eff > 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
            num = f"{'+' if eff > 0 else '−'}{_round(eff)}"
            pos = f"left:calc(50% + {w:.1f}% + 8px)" if eff > 0 else f"right:calc(50% + {w:.1f}% + 8px)"
            rows.append(
                f'<div style="display:grid;grid-template-columns:minmax(150px,240px) 1fr;align-items:center;gap:8px;padding:5px 0" '
                f'data-driver="{esc(label)}" data-effect="{eff:.1f}">'
                f'<div><div style="font-size:14px;font-weight:600">{esc(label)}</div><div class="mt-caption">{esc(value)}</div></div>'
                f'<div style="position:relative;height:22px"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--mt-hairline)"></div>'
                f'<div style="position:absolute;top:4px;height:14px;border-radius:2px;background:var(--mt-{color});{side}"></div>'
                f'<div style="position:absolute;top:1px;font-size:14px;font-weight:600;color:var(--mt-{color});{pos}">{num}</div></div></div>'
            )
    legend = (
        '<div style="display:flex;gap:16px;justify-content:flex-end" class="mt-caption">'
        '<span><i style="display:inline-block;width:12px;height:10px;border-radius:2px;background:var(--mt-target)"></i> Lowers the peak</span>'
        '<span><i style="display:inline-block;width:12px;height:10px;border-radius:2px;background:var(--mt-warning)"></i> Raises the peak</span></div>'
    )
    return legend + "".join(rows)


def _round(x: float) -> int:
    x = abs(x)
    return int(5 * round(x / 5)) if x >= 10 else int(round(x))


def check_row(ok: bool, title: str, detail: str) -> str:
    color, ic, word = ("target", "circle-check", "agrees") if ok else ("critical", "circle-x", "disagrees")
    return (
        f'<div class="mt-check"><span style="color:var(--mt-{color})">{icon(ic, 18, word)}</span>'
        f'<div><b>{esc(title)}: {word}</b><span class="mt-caption">{esc(detail)}</span></div></div>'
    )


# --------------------------------------------------------------------------- triage table (Alert queue)


def triage_table(rows: Sequence[dict]) -> str:
    """rows: dicts with tier, pid, cohort, event, prob, prob_txt, conf, lead_frac, lead_txt, dq, age, critical."""
    head = "".join(
        f"<th scope='col'>{h}</th>"
        for h in ("Tier", "Patient", "Event", "Probability", "Confidence", "Lead time", "Data quality", "Age")
    )
    body = []
    for r in rows:
        tcolor = TIER_COLOR.get(r["tier"], "unknown")
        prob = (bar(r["prob"], tcolor) if r["prob"] is not None else bar(0, tcolor)) + esc(r["prob_txt"])
        lead = (bar(r["lead_frac"], "ink-muted") if r["lead_frac"] is not None else bar(0, "ink-muted")) + esc(
            r["lead_txt"]
        )
        conf_cls = ' class="low"' if r["conf"] is not None and r["conf"] < 0.5 else ""
        dq_cls = ' class="warn"' if r["dq"] is not None and r["dq"] < 0.9 else ""
        conf = "n/a" if r["conf"] is None else f"{r['conf']:.2f}"
        dq = "n/a" if r["dq"] is None else f"{r['dq']:.2f}"
        cls = ' class="critical"' if r["tier"] == "CRITICAL" else ""
        body.append(
            f"<tr{cls} data-alert='{esc(r['id'])}'><td>{tier_badge(r['tier'])}{' ↑' if r.get('escalated') else ''}</td>"
            f"<td><b>{esc(r['pid'])}</b><span class='muted'>{esc(r['cohort'])}</span></td>"
            f"<td>{esc(r['event'])}</td><td>{prob}</td><td{conf_cls}>{conf}</td><td>{lead}</td>"
            f"<td{dq_cls}>{dq}</td><td class='muted' style='margin:0'>{esc(r['age'])}</td></tr>"
        )
    return f'<table class="mt-table" aria-label="Alert queue"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def meal_chips(m) -> str:
    return "".join(
        f"<span class='mt-chip'>{lab} {v:.0f} g</span>"
        for lab, v in (("carbs", m.carbs_g), ("protein", m.protein_g), ("fat", m.fat_g), ("fibre", m.fibre_g))
    )
