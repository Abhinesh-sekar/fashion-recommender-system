"""
Cold-Start Analysis and Segmentation Module.
Evaluates recommendation performance across user activity cohorts (warm vs cold/low-activity)
and newly introduced items vs established catalog items.
"""

from typing import Dict, List, Tuple
import pandas as pd
from src.evaluation.metrics import evaluate_recommendations


class ColdStartEvaluator:
    """Evaluates cold-start behavior and performance differentials."""

    def __init__(self, train_transactions: pd.DataFrame):
        # Calculate historical activity counts per user and per item
        self.user_counts = train_transactions.groupby("customer_id")["article_id"].count().to_dict()
        self.item_counts = train_transactions.groupby("article_id")["price"].count().to_dict()

    def segment_users(
        self,
        actuals: Dict[str, List[str]],
        cold_threshold: int = 2,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits test users into:
        - cold_users: historical train transactions <= cold_threshold
        - warm_users: historical train transactions > cold_threshold
        """
        cold_actuals = {}
        warm_actuals = {}

        for cust_id, items in actuals.items():
            tx_count = self.user_counts.get(cust_id, 0)
            if tx_count <= cold_threshold:
                cold_actuals[cust_id] = items
            else:
                warm_actuals[cust_id] = items

        return cold_actuals, warm_actuals

    def evaluate_cohorts(
        self,
        actuals: Dict[str, List[str]],
        predictions: Dict[str, List[str]],
        cold_threshold: int = 2,
        k: int = 12,
    ) -> pd.DataFrame:
        """
        Computes MAP@K and HitRate@K for overall, warm, and cold user cohorts.
        """
        cold_actuals, warm_actuals = self.segment_users(actuals, cold_threshold=cold_threshold)

        overall_metrics = evaluate_recommendations(actuals, predictions, k=k)
        warm_metrics = evaluate_recommendations(warm_actuals, predictions, k=k)
        cold_metrics = evaluate_recommendations(cold_actuals, predictions, k=k)

        results = [
            {
                "cohort": "Overall",
                "user_count": len(actuals),
                f"map@{k}": overall_metrics[f"map@{k}"],
                f"hit_rate@{k}": overall_metrics[f"hit_rate@{k}"],
                f"precision@{k}": overall_metrics[f"precision@{k}"],
            },
            {
                "cohort": f"Warm (>{cold_threshold} past tx)",
                "user_count": len(warm_actuals),
                f"map@{k}": warm_metrics[f"map@{k}"],
                f"hit_rate@{k}": warm_metrics[f"hit_rate@{k}"],
                f"precision@{k}": warm_metrics[f"precision@{k}"],
            },
            {
                "cohort": f"Cold (≤{cold_threshold} past tx)",
                "user_count": len(cold_actuals),
                f"map@{k}": cold_metrics[f"map@{k}"],
                f"hit_rate@{k}": cold_metrics[f"hit_rate@{k}"],
                f"precision@{k}": cold_metrics[f"precision@{k}"],
            },
        ]
        return pd.DataFrame(results)
