"""
Multi-Source Candidate Generation Orchestrator.
Blends user repurchase history, co-purchase graph retrieval, category affinity,
and temporal popularity to generate a high-recall candidate pool (50-80 items / user).
"""

from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
from src.retrieval.co_purchase import CoPurchaseRetriever
from src.retrieval.popularity import PopularityRetriever


class MultiSourceCandidateGenerator:
    """Orchestrates Stage 1 candidate generation across multiple retrieval channels."""

    def __init__(self, n_candidates: int = 60):
        self.n_candidates = n_candidates
        self.pop_retriever = PopularityRetriever(recent_days=14)
        self.co_retriever = CoPurchaseRetriever(min_co_occurrences=2)
        self.user_history: Dict[str, List[str]] = defaultdict(list)
        self.customer_affinity: Dict[str, str] = {}

    def fit(
        self,
        transactions_df: pd.DataFrame,
        articles_df: Optional[pd.DataFrame] = None,
        customers_df: Optional[pd.DataFrame] = None,
    ) -> "MultiSourceCandidateGenerator":
        self.pop_retriever.fit(transactions_df, articles_df)
        self.co_retriever.fit(transactions_df)

        # Store user purchase history in chronological order
        df = transactions_df.sort_values(by="t_dat")
        for cust_id, group in df.groupby("customer_id"):
            self.user_history[str(cust_id)] = group["article_id"].tail(20).tolist()

        if customers_df is not None and "primary_interest" in customers_df.columns:
            for _, row in customers_df.iterrows():
                self.customer_affinity[str(row["customer_id"])] = str(row["primary_interest"])

        return self

    def generate_candidates_for_user(self, customer_id: str) -> List[str]:
        """Generate deduplicated candidates for a single customer."""
        candidates = []
        seen = set()

        # Channel 1: Recent Repurchases (High Precision)
        past_items = self.user_history.get(customer_id, [])
        for item in reversed(past_items[-10:]):
            if item not in seen:
                candidates.append(item)
                seen.add(item)

        # Channel 2: Co-purchase Collaborative Filtering (High Serendipity)
        if past_items:
            co_items = self.co_retriever.retrieve_for_items(past_items[-5:], n_candidates=25)
            for item in co_items:
                if item not in seen:
                    candidates.append(item)
                    seen.add(item)

        # Channel 3: Customer Primary Interest / Category Popularity
        affinity = self.customer_affinity.get(customer_id)
        cat_pop = self.pop_retriever.retrieve(n_candidates=20, category=affinity)
        for item in cat_pop:
            if item not in seen:
                candidates.append(item)
                seen.add(item)

        # Channel 4: Global Popularity Backfill
        global_pop = self.pop_retriever.retrieve(n_candidates=self.n_candidates)
        for item in global_pop:
            if item not in seen:
                candidates.append(item)
                seen.add(item)
            if len(candidates) >= self.n_candidates:
                break

        return candidates[:self.n_candidates]

    def generate_candidate_dataframe(self, customer_ids: List[str]) -> pd.DataFrame:
        """Generate candidate pairs (customer_id, article_id) for a list of customers."""
        rows = []
        for cust_id in customer_ids:
            cand_items = self.generate_candidates_for_user(cust_id)
            for rank_pos, item in enumerate(cand_items, 1):
                rows.append({
                    "customer_id": cust_id,
                    "article_id": item,
                    "retrieval_rank": rank_pos,
                })
        return pd.DataFrame(rows)
