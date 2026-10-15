"""Unit tests for evaluation metrics (MAP@12, precision, recall, hit rate)."""

import pytest
from src.evaluation.metrics import (
    average_precision_at_k,
    mean_average_precision_at_k,
    precision_recall_hitrate_at_k,
    evaluate_recommendations,
)


def test_average_precision_at_k_perfect():
    actual = ["A", "B", "C"]
    predicted = ["A", "B", "C", "D", "E"]
    # Hits at positions 1, 2, 3: precisions 1/1, 2/2, 3/3 = 1.0. denominator min(3, 12) = 3 -> AP = 1.0
    assert average_precision_at_k(actual, predicted, k=12) == pytest.approx(1.0)


def test_average_precision_at_k_partial():
    actual = ["A", "B"]
    predicted = ["X", "A", "Y", "B"]
    # Hits at pos 2 (P=1/2), pos 4 (P=2/4=0.5). sum = 1.0. denominator min(2, 12) = 2 -> AP = 0.5
    assert average_precision_at_k(actual, predicted, k=12) == pytest.approx(0.5)


def test_average_precision_at_k_empty():
    assert average_precision_at_k([], ["A", "B"], k=12) == 0.0
    assert average_precision_at_k(["A"], [], k=12) == 0.0


def test_mean_average_precision_at_k():
    actuals = {
        "user1": ["A", "B"],
        "user2": ["C"],
    }
    predictions = {
        "user1": ["A", "B"],      # AP = 1.0
        "user2": ["X", "Y", "Z"], # AP = 0.0
    }
    map_score = mean_average_precision_at_k(actuals, predictions, k=12)
    assert map_score == pytest.approx(0.5)


def test_precision_recall_hitrate_at_k():
    actuals = {"u1": ["A", "B"]}
    preds = {"u1": ["A", "X", "Y"]}
    res = precision_recall_hitrate_at_k(actuals, preds, k=2)
    assert res["precision@2"] == pytest.approx(0.5) # 1 hit / 2 preds
    assert res["recall@2"] == pytest.approx(0.5)    # 1 hit / 2 actual
    assert res["hit_rate@2"] == pytest.approx(1.0)  # at least 1 hit


def test_evaluate_recommendations():
    actuals = {"u1": ["item_1", "item_2"]}
    predictions = {"u1": ["item_1", "item_3"]}
    result = evaluate_recommendations(actuals, predictions, k=12)
    assert "map@12" in result
    assert "hit_rate@12" in result
    assert result["n_evaluated_users"] == 1
