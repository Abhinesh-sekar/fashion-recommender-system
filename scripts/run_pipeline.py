"""
End-to-End Pipeline Execution Script.
Executes:
1. Data loading and ground-truth extraction
2. Baseline Popularity evaluation (MAP@12)
3. Stage 1 Multi-source Candidate Retrieval
4. Feature Engineering and LightGBM Re-Ranker training
5. Stage 2 Test Inference & Personalized Ranking (Top-12)
6. Comprehensive Evaluation & Cold-Start Analysis
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.candidate_generator import MultiSourceCandidateGenerator
from src.retrieval.popularity import PopularityRetriever
from src.ranking.feature_builder import ReRankingFeatureBuilder
from src.ranking.lgbm_ranker import LightGBMReRanker
from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.cold_start import ColdStartEvaluator


def load_datasets(data_dir: str = "data/raw"):
    print("[1/6] Loading parquet datasets...")
    train_df = pd.read_parquet(os.path.join(data_dir, "transactions_train.parquet"))
    test_df = pd.read_parquet(os.path.join(data_dir, "transactions_test.parquet"))
    articles_df = pd.read_parquet(os.path.join(data_dir, "articles.parquet"))
    customers_df = pd.read_parquet(os.path.join(data_dir, "customers.parquet"))

    # Convert t_dat to datetime
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    test_df["t_dat"] = pd.to_datetime(test_df["t_dat"])

    print(f"  Train: {len(train_df):,} transactions ({train_df['customer_id'].nunique():,} users)")
    print(f"  Test:  {len(test_df):,} transactions ({test_df['customer_id'].nunique():,} users)")
    print(f"  Catalog: {len(articles_df):,} articles, {len(customers_df):,} customers")
    return train_df, test_df, articles_df, customers_df


def extract_ground_truth(test_df: pd.DataFrame) -> dict:
    """Extract dict of {customer_id: [ground_truth_article_ids]}."""
    return test_df.groupby("customer_id")["article_id"].unique().apply(list).to_dict()


def run_pipeline():
    start_time = time.time()
    train_df, test_df, articles_df, customers_df = load_datasets()
    ground_truth = extract_ground_truth(test_df)
    test_customers = list(ground_truth.keys())

    # -------------------------------------------------------------
    # 2. Baseline Evaluation: Recent Global Popularity
    # -------------------------------------------------------------
    print("\n[2/6] Evaluating Baseline: Recent Global Popularity Top-12...")
    pop_retriever = PopularityRetriever(recent_days=7)
    pop_retriever.fit(train_df, articles_df)
    baseline_top12 = pop_retriever.retrieve(n_candidates=12)
    baseline_preds = {cust_id: baseline_top12 for cust_id in test_customers}

    baseline_metrics = evaluate_recommendations(ground_truth, baseline_preds, k=12)
    print(f"  Baseline Popularity MAP@12:    {baseline_metrics['map@12']:.5f}")
    print(f"  Baseline Popularity HitRate@12:{baseline_metrics['hit_rate@12']*100:.2f}%")
    print(f"  Baseline Popularity Recall@12: {baseline_metrics['recall@12']*100:.2f}%")

    # -------------------------------------------------------------
    # 3. Construct Training Data for Stage 2 Re-Ranker
    # -------------------------------------------------------------
    print("\n[3/6] Generating Training Candidates & Supervised Target Labels...")
    # Time-split: Hold out last 7 days of training as local validation / training targets
    train_split_date = train_df["t_dat"].max() - pd.Timedelta(days=7)
    train_history = train_df[train_df["t_dat"] < train_split_date].copy()
    train_targets = train_df[train_df["t_dat"] >= train_split_date].copy()

    # Candidate generator fitted on train_history
    cand_gen_train = MultiSourceCandidateGenerator(n_candidates=40)
    cand_gen_train.fit(train_history, articles_df, customers_df)

    # Sample 2,000 active target customers for fast, high-quality training
    target_custs = train_targets["customer_id"].unique()[:2000].tolist()
    train_candidate_pairs = cand_gen_train.generate_candidate_dataframe(target_custs)

    # Positive targets map: set of (customer_id, article_id)
    positives = set(zip(train_targets["customer_id"], train_targets["article_id"]))
    train_candidate_pairs["label"] = [
        1 if (c, a) in positives else 0
        for c, a in zip(train_candidate_pairs["customer_id"], train_candidate_pairs["article_id"])
    ]
    pos_count = train_candidate_pairs["label"].sum()
    print(f"  Generated {len(train_candidate_pairs):,} training pairs ({pos_count:,} positives, {pos_count/len(train_candidate_pairs)*100:.2f}% positive rate)")

    # -------------------------------------------------------------
    # 4. Feature Extraction & LightGBM Model Training
    # -------------------------------------------------------------
    print("\n[4/6] Engineering Features & Training Stage 2 LightGBM Re-Ranker...")
    feature_builder = ReRankingFeatureBuilder()
    feature_builder.fit(train_history, articles_df, customers_df)
    train_features = feature_builder.build_features(train_candidate_pairs)

    ranker = LightGBMReRanker()
    ranker.fit(train_features, label_col="label")

    # Show top 5 features
    top_features = ranker.get_feature_importances()
    print("  Top 5 Feature Importances (Gain):")
    for _, row in top_features.head(5).iterrows():
        print(f"    - {row['feature']:<25}: {row['gain_importance']:,.1f}")

    # -------------------------------------------------------------
    # 5. Test Inference: Candidates + Re-Ranking for Test Users
    # -------------------------------------------------------------
    print(f"\n[5/6] Generating Test Candidates & Scoring for {len(test_customers):,} Test Customers...")
    # Re-fit candidate generator & feature builder on full train_df
    cand_gen_full = MultiSourceCandidateGenerator(n_candidates=60)
    cand_gen_full.fit(train_df, articles_df, customers_df)

    test_candidate_pairs = cand_gen_full.generate_candidate_dataframe(test_customers)
    print(f"  Generated {len(test_candidate_pairs):,} test candidate pairs.")

    feature_builder_full = ReRankingFeatureBuilder()
    feature_builder_full.fit(train_df, articles_df, customers_df)
    test_features = feature_builder_full.build_features(test_candidate_pairs)

    model_preds = ranker.predict_rankings(test_features, top_k=12)

    # -------------------------------------------------------------
    # 6. Evaluation & Cold-Start Analysis
    # -------------------------------------------------------------
    print("\n[6/6] Computing Final Evaluation Metrics & Cohort Breakdown...")
    model_metrics = evaluate_recommendations(ground_truth, model_preds, k=12)

    print("\n" + "=" * 65)
    print("           MODEL BENCHMARK COMPARISON (Top-12)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Popularity Baseline':<20} | {'Two-Stage LightGBM':<20}")
    print("-" * 65)
    print(f"{'MAP@12':<20} | {baseline_metrics['map@12']:<20.5f} | {model_metrics['map@12']:<20.5f}")
    print(f"{'HitRate@12':<20} | {baseline_metrics['hit_rate@12']*100:<19.2f}% | {model_metrics['hit_rate@12']*100:<19.2f}%")
    print(f"{'Recall@12':<20} | {baseline_metrics['recall@12']*100:<19.2f}% | {model_metrics['recall@12']*100:<19.2f}%")
    print(f"{'Precision@12':<20} | {baseline_metrics['precision@12']*100:<19.2f}% | {model_metrics['precision@12']*100:<19.2f}%")
    print("=" * 65)

    map_rel_lift = (model_metrics['map@12'] - baseline_metrics['map@12']) / baseline_metrics['map@12'] * 100
    print(f"\n>> Relative MAP@12 Improvement over baseline: +{map_rel_lift:.1f}%\n")

    # Cold Start Analysis
    cold_evaluator = ColdStartEvaluator(train_df)
    cohort_df = cold_evaluator.evaluate_cohorts(ground_truth, model_preds, cold_threshold=5, k=12)
    print("--- Cold-Start / Activity Cohort Performance Breakdown ---")
    print(cohort_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.1f}s.")
    return baseline_metrics, model_metrics, cohort_df


if __name__ == "__main__":
    run_pipeline()
