"""
Evaluation Metrics for Fashion Recommendation System.
Implements Kaggle H&M Official Metric: Mean Average Precision at 12 (MAP@12),
along with Precision@K, Recall@K, and HitRate@K.
"""

from typing import Dict, List, Set
import numpy as np
import pandas as pd


def average_precision_at_k(actual: List[str], predicted: List[str], k: int = 12) -> float:
    """
    Computes Average Precision at K for a single customer.
    
    Formula:
        AP@K = sum_{i=1..min(K, len(predicted))} (P(i) * rel(i)) / min(K, len(actual))
        where rel(i) = 1 if predicted[i-1] in actual else 0.
    """
    if not actual:
        return 0.0

    actual_set: Set[str] = set(actual)
    pred_k = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(pred_k):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    denominator = min(len(actual_set), k)
    return score / denominator if denominator > 0 else 0.0


def mean_average_precision_at_k(
    actuals: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    k: int = 12,
) -> float:
    """
    Computes MAP@K across all customers in actuals.
    """
    if not actuals:
        return 0.0

    ap_scores = [
        average_precision_at_k(actual=actuals[cust_id], predicted=predictions.get(cust_id, []), k=k)
        for cust_id in actuals
    ]
    return float(np.mean(ap_scores))


def precision_recall_hitrate_at_k(
    actuals: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    k: int = 12,
) -> Dict[str, float]:
    """
    Calculates Precision@K, Recall@K, and HitRate@K across all test customers.
    """
    if not actuals:
        return {"precision@k": 0.0, "recall@k": 0.0, "hit_rate@k": 0.0}

    precisions = []
    recalls = []
    hits = []

    for cust_id, actual in actuals.items():
        actual_set = set(actual)
        preds = predictions.get(cust_id, [])[:k]
        n_hits = len(set(preds).intersection(actual_set))

        precisions.append(n_hits / float(k))
        recalls.append(n_hits / float(len(actual_set)) if actual_set else 0.0)
        hits.append(1.0 if n_hits > 0 else 0.0)

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"hit_rate@{k}": float(np.mean(hits)),
    }


def evaluate_recommendations(
    actuals: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    k: int = 12,
) -> Dict[str, float]:
    """
    Comprehensive evaluation dictionary returning MAP@K, Precision@K, Recall@K, HitRate@K.
    """
    map_k = mean_average_precision_at_k(actuals, predictions, k=k)
    other_metrics = precision_recall_hitrate_at_k(actuals, predictions, k=k)
    return {
        f"map@{k}": map_k,
        **other_metrics,
        "n_evaluated_users": len(actuals),
    }
