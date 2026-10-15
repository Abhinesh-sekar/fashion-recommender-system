"""
Feature Engineering Engine for Two-Stage Re-Ranking.
Enriches candidate pairs (customer_id, article_id) with user behavior,
article velocity, and user-item interaction signals.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ReRankingFeatureBuilder:
    """Builds tabular features for Stage 2 LightGBM re-ranker."""

    def __init__(self):
        self.user_stats: pd.DataFrame = pd.DataFrame()
        self.item_stats: pd.DataFrame = pd.DataFrame()
        self.user_item_history: Dict[Tuple[str, str], int] = {}
        self.user_categories: Dict[str, str] = {}
        self.article_metadata: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        transactions_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        customers_df: Optional[pd.DataFrame] = None,
    ) -> "ReRankingFeatureBuilder":
        print("[FeatureBuilder] Computing user and item historical features...")
        df = transactions_df.copy()
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        max_date = df["t_dat"].max()

        # 1. User Features
        user_grp = df.groupby("customer_id")
        self.user_stats = pd.DataFrame({
            "user_tx_count": user_grp["article_id"].count(),
            "user_unique_items": user_grp["article_id"].nunique(),
            "user_avg_price": user_grp["price"].mean(),
            "user_std_price": user_grp["price"].std().fillna(0.0),
            "user_days_since_last": (max_date - user_grp["t_dat"].max()).dt.days,
        }).reset_index()

        # 2. Item Features
        item_grp = df.groupby("article_id")
        last_7d = df[df["t_dat"] >= max_date - pd.Timedelta(days=7)]
        recent_sales = last_7d.groupby("article_id")["price"].count().rename("item_sales_last_7d")

        self.item_stats = pd.DataFrame({
            "item_total_sales": item_grp["price"].count(),
            "item_unique_buyers": item_grp["customer_id"].nunique(),
            "item_avg_price": item_grp["price"].mean(),
        }).reset_index()

        self.item_stats = self.item_stats.merge(recent_sales, on="article_id", how="left")
        self.item_stats["item_sales_last_7d"] = self.item_stats["item_sales_last_7d"].fillna(0)

        # 3. User-Item Interaction Map
        for (c_id, a_id), count in df.groupby(["customer_id", "article_id"]).size().items():
            self.user_item_history[(str(c_id), str(a_id))] = int(count)

        # 4. Article metadata & customer affinity
        self.article_metadata = articles_df.copy()
        if customers_df is not None and "primary_interest" in customers_df.columns:
            self.user_categories = dict(zip(customers_df["customer_id"], customers_df["primary_interest"]))

        return self

    def build_features(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich candidate pairs with interaction, user, and item features."""
        df = candidates_df.copy()

        # Merge user features
        df = df.merge(self.user_stats, on="customer_id", how="left")
        df["user_tx_count"] = df["user_tx_count"].fillna(0)
        df["user_avg_price"] = df["user_avg_price"].fillna(0.02)
        df["user_days_since_last"] = df["user_days_since_last"].fillna(999)

        # Merge item features
        df = df.merge(self.item_stats, on="article_id", how="left")
        df["item_total_sales"] = df["item_total_sales"].fillna(0)
        df["item_sales_last_7d"] = df["item_sales_last_7d"].fillna(0)
        df["item_avg_price"] = df["item_avg_price"].fillna(0.02)

        # Merge article metadata
        art_cols = ["article_id", "index_group_name", "product_type_name", "base_price"]
        df = df.merge(self.article_metadata[art_cols], on="article_id", how="left")

        # 5. Cross & Interaction Features
        # Has user bought this item before?
        df["user_item_past_purchases"] = [
            self.user_item_history.get((c, a), 0) for c, a in zip(df["customer_id"], df["article_id"])
        ]
        df["user_has_bought_before"] = (df["user_item_past_purchases"] > 0).astype(int)

        # Price difference from user's typical price
        df["price_diff_from_user_avg"] = np.abs(df["base_price"] - df["user_avg_price"])

        # Category match with user primary preference
        df["user_primary_affinity"] = df["customer_id"].map(self.user_categories)
        df["is_category_match"] = (df["index_group_name"] == df["user_primary_affinity"]).astype(int)

        # Relative rank feature
        df["inv_retrieval_rank"] = 1.0 / df["retrieval_rank"]

        return df
