from datetime import UTC, datetime, timedelta

import pytest

from src.contracts import Forecast
from src.serving import alerts as al
from src.serving import backend, mock_data
from src.serving.mock_predict import mock_thresholds

NOW = datetime(2026, 11, 19, 14, 0, tzinfo=UTC)
THR = mock_thresholds()


def fc(spike=0.0, hypo=0.0, conf=0.8, g=(120, 125, 130), rule=False, traj=None):
    return Forecast(glucose_30=g, glucose_60=g, glucose_120=g, spike_prob=spike, hypo_prob=hypo, confidence=conf,
                    detector_flags={"quantile": False, "classifier": False, "rule_net": rule},
                    trajectory=traj, model_version="test-1")


def inp(f, cohort="t2d", g_now=120.0, dq=1.0, stale=0.0, pid="P-100", now=NOW):
    return al.AlertInput(pid, cohort, now, f, g_now, dq, stale)


def tiers(cands):
    return {c.event: c.tier for c in cands}


def test_no_alert_when_quiet():
    assert al.evaluate(inp(fc()), THR) == []


def test_current_hypo_is_critical():
    assert tiers(al.evaluate(inp(fc(), g_now=65), THR))["hypo"] == al.Tier.CRITICAL


def test_q10_below_70_within_30_is_critical():
    f = fc(g=(65, 85, 100))
    assert tiers(al.evaluate(inp(f, g_now=95), THR))["hypo"] == al.Tier.CRITICAL


def test_rule_net_critical_maps_to_event():
    assert tiers(al.evaluate(inp(fc(rule=True), g_now=85), THR))["hypo"] == al.Tier.CRITICAL
    assert tiers(al.evaluate(inp(fc(rule=True), g_now=320), THR))["spike"] == al.Tier.CRITICAL


def test_warning_vs_watch_gating():
    assert tiers(al.evaluate(inp(fc(spike=0.9, conf=0.8)), THR))["spike"] == al.Tier.WARNING
    assert tiers(al.evaluate(inp(fc(spike=0.9, conf=0.2)), THR))["spike"] == al.Tier.WATCH


@pytest.mark.parametrize("conf", [0.0, 0.05, 0.3, 0.49])
def test_hypo_never_suppressed_by_low_confidence(conf):
    t = tiers(al.evaluate(inp(fc(hypo=0.6, conf=conf)), THR))
    assert t.get("hypo", al.Tier.NONE) >= al.Tier.WATCH


def test_stratum_thresholds_differ():
    f = fc(spike=0.47)
    assert "spike" in tiers(al.evaluate(inp(f, cohort="t2d"), THR))  # 0.40
    assert "spike" not in tiers(al.evaluate(inp(f, cohort="healthy"), THR))  # 0.50


def test_stale_is_unknown_never_low_risk():
    t = tiers(al.evaluate(inp(fc(), cohort="healthy", g_now=None, stale=45), THR))
    assert t["unknown"] == al.Tier.WARNING
    t = tiers(al.evaluate(inp(fc(), cohort="healthy", g_now=None, stale=45), THR, recently_high_risk=True))
    assert t["unknown"] == al.Tier.CRITICAL
    t = tiers(al.evaluate(inp(fc(), cohort="t2d", g_now=None, stale=float("inf")), THR))
    assert t["unknown"] == al.Tier.CRITICAL


def test_lead_time_from_trajectory_and_buckets():
    traj = [{"t_min": m, "q10": 150 + m, "q50": 160 + m, "q90": 170 + m} for m in range(5, 125, 5)]
    c = al.evaluate(inp(fc(spike=0.9, g=(190, 220, 230), traj=traj)), THR)[0]
    assert c.lead_min == 25 and c.lead_label == "~25 min"
    no_traj = Forecast(glucose_30=(140, 150, 160), glucose_60=(165, 185, 205), glucose_120=(150, 170, 190),
                       spike_prob=0.9, hypo_prob=0.0, confidence=0.8)
    c = al.evaluate(inp(no_traj), THR)[0]
    assert c.lead_label == "≤60 min"  # bucketed: no false precision without a trajectory


def cand(tier=al.Tier.WARNING, event="spike", pid="P-100", prob=0.8):
    return al.Candidate(pid, "t2d", event, tier, prob, 0.8, 30, "~30 min", ("classifier",), 1.0, "m", "r")


def test_episode_dedupe_updates_in_place():
    book = al.AlertBook()
    book.update(NOW, [cand(prob=0.6)])
    book.update(NOW + timedelta(minutes=15), [cand(prob=0.9)])
    assert len(book.history) == 1 and book.history[0].probability == 0.9 and book.history[0].n_evals == 2


def test_hysteresis_then_cooldown_suppresses_watch_only():
    book = al.AlertBook()
    t = NOW
    book.update(t, [cand(al.Tier.WATCH)])
    book.update(t := t + timedelta(minutes=15), [], evaluated=["P-100"])
    assert book.active  # absent once: still open (hysteresis)
    book.update(t := t + timedelta(minutes=15), [], evaluated=["P-100"])
    assert not book.active and book.history[0].resolved_at == t
    book.update(t := t + timedelta(minutes=15), [cand(al.Tier.WATCH)])
    assert len(book.history) == 1 and book.suppressed == 1
    book.update(t := t + timedelta(minutes=15), [cand(al.Tier.WARNING)])
    assert len(book.history) == 2  # WARNING fires despite cooldown


def test_critical_hypo_bypasses_cooldown():
    book = al.AlertBook()
    key = ("P-100", "hypo")
    book.cooldown_until[key] = NOW + timedelta(hours=1)
    book.update(NOW, [cand(al.Tier.CRITICAL, event="hypo")])
    assert book.active[key].tier == al.Tier.CRITICAL


def test_cooldown_expires():
    book = al.AlertBook()
    book.cooldown_until[("P-100", "spike")] = NOW + timedelta(minutes=60)
    book.update(NOW + timedelta(minutes=61), [cand(al.Tier.WATCH)])
    assert len(book.history) == 1


def test_watch_escalates_after_two_evaluations():
    book = al.AlertBook()
    book.update(NOW, [cand(al.Tier.WATCH)])
    assert book.active[("P-100", "spike")].tier == al.Tier.WATCH
    book.update(NOW + timedelta(minutes=15), [cand(al.Tier.WATCH)])
    a = book.active[("P-100", "spike")]
    assert a.tier == al.Tier.WARNING and a.escalated


def test_queue_sorting():
    a = [al.Alert(str(i), "P", "t2d", "spike", t, p, 0.8, lead, "", (), 1.0, "m", "r", NOW, NOW)
         for i, (t, lead, p) in enumerate([(al.Tier.WARNING, 30, 0.5), (al.Tier.CRITICAL, 60, 0.9),
                                           (al.Tier.WARNING, 10, 0.4), (al.Tier.WARNING, 10, 0.9)])]
    assert [x.id for x in al.sort_queue(a)] == ["1", "3", "2", "0"]


def test_burden_arithmetic():
    book = al.AlertBook()
    book.update(NOW, [cand(al.Tier.WARNING, pid="A"), cand(al.Tier.CRITICAL, "hypo", pid="B"),
                      cand(al.Tier.WATCH, pid="C")])
    b = al.burden(book, n_patients=3, window_hours=24)
    assert (b.episodes, b.actionable, b.critical) == (3, 2, 1)
    assert b.per_patient_day == 1.0 and b.critical_share == pytest.approx(1 / 3, abs=1e-3)
    assert b.within_budget


def test_replay_on_mock_day():
    times = [t for t in mock_data.replay_times(30) if t <= mock_data.DEMO_NOW]
    snaps = al.replay(times, backend.list_patients(), backend.get_state,
                      lambda s: backend.predict(s, explain=False), THR)
    last = snaps[-1]
    by = {(a.patient_id, a.event): a for a in last.queue}
    assert by[("P-011", "hypo")].tier == al.Tier.CRITICAL
    assert by[("P-007", "unknown")].tier >= al.Tier.WARNING
    assert by[("P-010", "spike")].tier >= al.Tier.WARNING
    assert last.burden.per_patient_day <= al.AlertConfig().budget_per_patient_day
