"""
LightGBM Second-Stage Re-Ranker.
Trains on candidate pool features with binary cross-entropy / LambdaMART ranking objective,
predicts personalized purchase probabilities, and outputs top-12 recommendations per customer.
"""

from typing import Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "retrieval_rank",
    "inv_retrieval_rank",
    "user_tx_count",
    "user_avg_price",
    "user_days_since_last",
    "item_total_sales",
    "item_sales_last_7d",
    "item_avg_price",
    "user_item_past_purchases",
    "user_has_bought_before",
    "price_diff_from_user_avg",
    "is_category_match",
]


class LightGBMReRanker:
    """Stage 2 Ranking Model."""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "n_estimators": 200,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_cols = FEATURE_COLUMNS

    def fit(self, train_features_df: pd.DataFrame, label_col: str = "label") -> "LightGBMReRanker":
        print(f"[ReRanker] Training LightGBM on {len(train_features_df):,} candidate pairs...")
        X = train_features_df[self.feature_cols]
        y = train_features_df[label_col]
        self.model.fit(X, y)
        print("[ReRanker] Model training completed.")
        return self

    def predict_rankings(
        self, candidates_features_df: pd.DataFrame, top_k: int = 12
    ) -> Dict[str, List[str]]:
        """
        Score candidate pairs and return top_k recommended article IDs per customer.
        """
        df = candidates_features_df.copy()
        X = df[self.feature_cols]
        df["score"] = self.model.predict_proba(X)[:, 1]

        # Rank descending per customer
        df = df.sort_values(by=["customer_id", "score"], ascending=[True, False])
        
        recommendations = {}
        for cust_id, group in df.groupby("customer_id"):
            recommendations[str(cust_id)] = group["article_id"].head(top_k).tolist()

        return recommendations

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": self.feature_cols, "gain_importance": imp}).sort_values(
            by="gain_importance", ascending=False
        ).reset_index(drop=True)
