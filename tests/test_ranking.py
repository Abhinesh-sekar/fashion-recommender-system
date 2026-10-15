"""Unit tests for Stage 2 Feature Builder and LightGBM Re-Ranker."""

import pandas as pd
import pytest
from src.ranking.feature_builder import ReRankingFeatureBuilder
from src.ranking.lgbm_ranker import LightGBMReRanker


@pytest.fixture
def dummy_data():
    transactions = pd.DataFrame([
        {"t_dat": "2024-06-01", "customer_id": "u1", "article_id": "a1", "price": 0.04},
        {"t_dat": "2024-06-02", "customer_id": "u1", "article_id": "a2", "price": 0.06},
        {"t_dat": "2024-06-03", "customer_id": "u2", "article_id": "a1", "price": 0.04},
        {"t_dat": "2024-06-04", "customer_id": "u3", "article_id": "a3", "price": 0.02},
    ])
    articles = pd.DataFrame([
        {"article_id": "a1", "index_group_name": "Ladieswear", "product_type_name": "Dress", "base_price": 0.04},
        {"article_id": "a2", "index_group_name": "Ladieswear", "product_type_name": "Blouse", "base_price": 0.06},
        {"article_id": "a3", "index_group_name": "Menswear", "product_type_name": "Pants", "base_price": 0.02},
    ])
    customers = pd.DataFrame([
        {"customer_id": "u1", "primary_interest": "Ladieswear"},
        {"customer_id": "u2", "primary_interest": "Ladieswear"},
        {"customer_id": "u3", "primary_interest": "Menswear"},
    ])
    return transactions, articles, customers


def test_feature_builder_and_ranker(dummy_data):
    transactions, articles, customers = dummy_data

    builder = ReRankingFeatureBuilder()
    builder.fit(transactions, articles, customers)

    candidates = pd.DataFrame([
        {"customer_id": "u1", "article_id": "a1", "retrieval_rank": 1},
        {"customer_id": "u1", "article_id": "a3", "retrieval_rank": 2},
        {"customer_id": "u2", "article_id": "a2", "retrieval_rank": 1},
        {"customer_id": "u2", "article_id": "a3", "retrieval_rank": 2},
        {"customer_id": "u3", "article_id": "a3", "retrieval_rank": 1},
        {"customer_id": "u3", "article_id": "a1", "retrieval_rank": 2},
    ])

    features_df = builder.build_features(candidates)
    assert "user_tx_count" in features_df.columns
    assert "item_total_sales" in features_df.columns
    assert "price_diff_from_user_avg" in features_df.columns
    assert "is_category_match" in features_df.columns

    # Check user u1 has bought a1 before
    row_u1_a1 = features_df[(features_df["customer_id"] == "u1") & (features_df["article_id"] == "a1")].iloc[0]
    assert row_u1_a1["user_has_bought_before"] == 1
    row_u1_a3 = features_df[(features_df["customer_id"] == "u1") & (features_df["article_id"] == "a3")].iloc[0]
    assert row_u1_a3["user_has_bought_before"] == 0

    # Fit LightGBM ranker on dummy labels
    features_df["label"] = [1, 0, 1, 0, 1, 0]
    ranker = LightGBMReRanker(params={"n_estimators": 10, "min_child_samples": 1})
    ranker.fit(features_df, label_col="label")

    rankings = ranker.predict_rankings(features_df, top_k=2)
    assert "u1" in rankings
    assert len(rankings["u1"]) == 2
