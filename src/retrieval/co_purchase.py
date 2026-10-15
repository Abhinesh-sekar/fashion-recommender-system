"""
Item-Item Collaborative Filtering & Co-Purchase Graph.
Builds co-occurrence graph of articles purchased together by identical customers.
Retrieves items complementary or frequently bought alongside a user's recent purchases.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


class CoPurchaseRetriever:
    """Item-Item co-purchase candidate generator."""

    def __init__(self, min_co_occurrences: int = 2):
        self.min_co_occurrences = min_co_occurrences
        self.co_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.top_co_items: Dict[str, List[Tuple[str, int]]] = {}

    def fit(self, transactions_df: pd.DataFrame) -> "CoPurchaseRetriever":
        print("[CoPurchase] Building item-item co-occurrence graph...")
        # Group articles purchased by customer
        cust_articles = (
            transactions_df.groupby("customer_id")["article_id"]
            .apply(lambda s: list(set(s)))
            .tolist()
        )

        for basket in cust_articles:
            if len(basket) < 2:
                continue
            # Increment co-occurrence pairs
            for i in range(len(basket)):
                a = basket[i]
                for j in range(i + 1, len(basket)):
                    b = basket[j]
                    self.co_matrix[a][b] += 1
                    self.co_matrix[b][a] += 1

        # Cache top co-items per article
        for item, co_dict in self.co_matrix.items():
            filtered = [(other, count) for other, count in co_dict.items() if count >= self.min_co_occurrences]
            filtered.sort(key=lambda x: -x[1])
            self.top_co_items[item] = filtered[:30]

        print(f"[CoPurchase] Co-occurrence index built for {len(self.top_co_items):,} articles.")
        return self

    def retrieve_for_items(self, item_ids: List[str], n_candidates: int = 20) -> List[str]:
        """Retrieve top co-purchased items given a list of seed items."""
        candidate_scores = defaultdict(int)
        for item in item_ids:
            if item in self.top_co_items:
                for other_item, count in self.top_co_items[item]:
                    if other_item not in item_ids:
                        candidate_scores[other_item] += count

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [item for item, _ in sorted_candidates[:n_candidates]]

    def get_co_purchase_count(self, item_a: str, item_b: str) -> int:
        """Return co-purchase frequency between two items."""
        return self.co_matrix.get(item_a, {}).get(item_b, 0)
