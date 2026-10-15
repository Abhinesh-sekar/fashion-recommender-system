"""
Popularity-Based Candidate Retrieval.
Computes top popular items overall and segmented by category / index group.
Forms baseline candidate source and serves cold-start users.
"""

from typing import Dict, List, Optional
import pandas as pd


class PopularityRetriever:
    """Retrieves top-N most frequently purchased articles in recent history."""

    def __init__(self, recent_days: int = 14):
        self.recent_days = recent_days
        self.global_popular: List[str] = []
        self.category_popular: Dict[str, List[str]] = {}

    def fit(self, transactions_df: pd.DataFrame, articles_df: Optional[pd.DataFrame] = None) -> "PopularityRetriever":
        df = transactions_df.copy()
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        max_date = df["t_dat"].max()
        cutoff = max_date - pd.Timedelta(days=self.recent_days)

        recent_df = df[df["t_dat"] >= cutoff]
        if recent_df.empty:
            recent_df = df

        # Overall popular articles
        self.global_popular = (
            recent_df.groupby("article_id")["price"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # Category popular articles if articles_df provided
        if articles_df is not None:
            merged = recent_df.merge(articles_df[["article_id", "index_group_name"]], on="article_id", how="left")
            for cat, group in merged.groupby("index_group_name"):
                self.category_popular[str(cat)] = (
                    group.groupby("article_id")["price"]
                    .count()
                    .sort_values(ascending=False)
                    .index.tolist()
                )

        return self

    def retrieve(self, n_candidates: int = 12, category: Optional[str] = None) -> List[str]:
        """Retrieve top-N popular article IDs."""
        if category and category in self.category_popular:
            cat_list = self.category_popular[category]
            if len(cat_list) >= n_candidates:
                return cat_list[:n_candidates]
            # Backfill with global popular
            return (cat_list + [a for a in self.global_popular if a not in cat_list])[:n_candidates]
        return self.global_popular[:n_candidates]
