import pandas as pd

from dashboard.components import tables


def abl(rows):
    return pd.DataFrame(rows, columns=["variant", "metric", "value", "ci_low", "ci_high", "n_participants"])


def test_best_highlighted_only_without_ci_overlap():
    overlapping = abl([("a", "rmse_60", 30, 27, 33, 45), ("b", "rmse_60", 31, 28, 34, 45)])
    assert tables.best_without_overlap(overlapping, "rmse_60") is None
    separated = abl([("a", "rmse_60", 25, 23, 27, 45), ("b", "rmse_60", 31, 28, 34, 45)])
    assert tables.best_without_overlap(separated, "rmse_60") == "a"


def test_higher_is_better_for_recall():
    df = abl([("a", "recall_spike", 0.9, 0.86, 0.94, 45), ("b", "recall_spike", 0.6, 0.55, 0.65, 45)])
    assert tables.best_without_overlap(df, "recall_spike") == "a"


def test_pivot_keeps_order_and_n():
    df = abl([("film", "rmse_60", 31.8, 28.2, 35.4, 45), ("concat", "rmse_60", 32.7, 29, 36.4, 44)])
    piv = tables.ablation_pivot(df)
    assert list(piv["variant"]) == ["film", "concat"]
    assert list(piv["n_participants"]) == [45, 44]
    assert piv.loc[0, "rmse_60"] == "31.8 [28.2, 35.4]"


def test_limitations_cover_architecture_doc_items():
    text = " ".join(tables.LIMITATIONS).lower()
    for phrase in ("pairing is synthetic", "45 participants", "proxy", "not clinically validated", "glucose-only"):
        assert phrase in text
