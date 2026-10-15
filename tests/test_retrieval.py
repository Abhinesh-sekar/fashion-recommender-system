"""Unit tests for Stage 1 candidate retrieval modules."""

import pandas as pd
import pytest
from src.retrieval.popularity import PopularityRetriever
from src.retrieval.co_purchase import CoPurchaseRetriever
from src.retrieval.candidate_generator import MultiSourceCandidateGenerator


@pytest.fixture
def sample_transactions():
    return pd.DataFrame([
        {"t_dat": "2024-06-01", "customer_id": "u1", "article_id": "a1", "price": 0.03},
        {"t_dat": "2024-06-01", "customer_id": "u1", "article_id": "a2", "price": 0.05},
        {"t_dat": "2024-06-02", "customer_id": "u2", "article_id": "a1", "price": 0.03},
        {"t_dat": "2024-06-02", "customer_id": "u2", "article_id": "a3", "price": 0.02},
        {"t_dat": "2024-06-03", "customer_id": "u3", "article_id": "a2", "price": 0.05},
        {"t_dat": "2024-06-04", "customer_id": "u1", "article_id": "a1", "price": 0.03},
    ])


@pytest.fixture
def sample_articles():
    return pd.DataFrame([
        {"article_id": "a1", "index_group_name": "Ladieswear", "product_type_name": "Dress"},
        {"article_id": "a2", "index_group_name": "Ladieswear", "product_type_name": "Blouse"},
        {"article_id": "a3", "index_group_name": "Menswear", "product_type_name": "T-shirt"},
    ])


def test_popularity_retriever(sample_transactions, sample_articles):
    pop = PopularityRetriever(recent_days=30)
    pop.fit(sample_transactions, sample_articles)
    
    top_items = pop.retrieve(n_candidates=2)
    assert len(top_items) == 2
    # a1 was bought 3 times, a2 bought 2 times
    assert top_items[0] == "a1"
    assert top_items[1] == "a2"

    menswear = pop.retrieve(n_candidates=1, category="Menswear")
    assert menswear == ["a3"]


def test_co_purchase_retriever(sample_transactions):
    co = CoPurchaseRetriever(min_co_occurrences=1)
    co.fit(sample_transactions)

    # u1 bought a1 and a2 on 2024-06-01
    assert co.get_co_purchase_count("a1", "a2") == 1
    similar_to_a1 = co.retrieve_for_items(["a1"], n_candidates=5)
    assert "a2" in similar_to_a1 or "a3" in similar_to_a1


def test_multi_source_candidate_generator(sample_transactions, sample_articles):
    gen = MultiSourceCandidateGenerator(n_candidates=5)
    gen.fit(sample_transactions, sample_articles)

    cands = gen.generate_candidates_for_user("u1")
    assert len(cands) > 0
    # User u1 has purchase history, recent items should be included
    assert "a1" in cands or "a2" in cands

    df_cands = gen.generate_candidate_dataframe(["u1", "u2"])
    assert "customer_id" in df_cands.columns
    assert "article_id" in df_cands.columns
    assert "retrieval_rank" in df_cands.columns
