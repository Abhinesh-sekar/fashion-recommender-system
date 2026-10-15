# Two-Stage Fashion Recommender System 👗👠

A production-grade, two-stage personalized recommendation engine tailored for large-scale e-commerce and fashion retail transactions (modeled on the **Kaggle H&M Personalized Fashion Recommendations** benchmark). The architecture decouples **high-recall candidate retrieval** from **high-precision GBDT re-ranking**, optimizing the competition-standard **Mean Average Precision at 12 (MAP@12)**.

---

## 🚀 Key Highlights & Benchmark Results

Evaluated across **5,000 customers**, **1,000 catalog articles**, and **80,000 transactions** with rolling 7-week training and a 1-week temporal holdout:

| Model Architecture | MAP@12 | HitRate@12 | Recall@12 | Precision@12 |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** (Recent 7-Day Velocity) | 0.00382 | 4.29% | 1.37% | 0.36% |
| **Two-Stage LightGBM Re-Ranker** | **0.01084** | **14.44%** | **4.48%** | **1.26%** |
| **Relative Improvement** | **+183.7%** | **+236.6%** | **+227.0%** | **+250.0%** |

### Cold-Start User Segmentation (Activity Cohorts)
| Customer Cohort | Sample Size | MAP@12 | HitRate@12 | Precision@12 |
| :--- | :---: | :---: | :---: | :---: |
| **Overall** | 4,826 | 0.01084 | 14.44% | 1.26% |
| **Warm Users** (>5 past purchases) | 4,778 | 0.01093 | 14.50% | 1.27% |
| **Cold Users** (≤5 past purchases) | 48 | 0.00206 | 8.33% | 0.69% |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        TX[("Transactions Train (63.5k tx)")]
        ART[("Articles Catalog (1k SKUs)")]
        CUST[("Customers (5k shoppers)")]
    end

    subgraph Stage 1: Multi-Source Candidate Retrieval (Top 60 / user)
        C1["Channel 1: Recent Repurchases<br/>(High-Precision History)"]
        C2["Channel 2: Co-Purchase Graph<br/>(Item-Item Collaborative Filtering)"]
        C3["Channel 3: Category Affinity<br/>(Segmented Popularity)"]
        C4["Channel 4: Temporal Velocity<br/>(Global 7-Day Trend Backfill)"]
        MERGE["Deduplication & Union<br/>(Pool: 60 Candidates / Customer)"]
    end

    subgraph Stage 2: Feature Store & LightGBM Re-Ranker
        FEAT["Tabular Feature Engineering:<br/>• Price Diff from User Avg<br/>• User Tx Count & Velocity<br/>• Item 7-day Sales & Total Velocity<br/>• User-Item Interaction History<br/>• Category Affinity Match"]
        LGBM["LightGBM Ranker / GBDT Classifier<br/>(Predicts P(Purchase | user, candidate))"]
    end

    subgraph Output & Evaluation
        SORT["Score Descending & Cut at Top-12"]
        METRICS["Kaggle H&M Metric Suite:<br/>MAP@12, HitRate@12, Precision@12, Recall@12"]
    end

    TX --> C1 & C2 & C3 & C4
    ART --> C3 & C4
    CUST --> C1 & C3
    C1 & C2 & C3 & C4 --> MERGE
    MERGE --> FEAT
    TX & ART & CUST --> FEAT
    FEAT --> LGBM
    LGBM --> SORT
    SORT --> METRICS
```

---

## 📂 Repository Structure

```
fashion-recommender-system/
├── config/
│   └── config.yaml              # Hyperparameters, candidate pool sizing, date ranges
├── data/
│   ├── raw/                     # Parquet datasets (articles, customers, transactions)
│   └── sample_hm_data.py        # Reproducible synthetic H&M data generator
├── src/
│   ├── retrieval/
│   │   ├── popularity.py        # Global and category-segmented popularity retrieval
│   │   ├── co_purchase.py       # Item-Item co-occurrence matrix & graph CF
│   │   └── candidate_generator.py # Multi-channel candidate pool blender
│   ├── ranking/
│   │   ├── feature_builder.py   # User, item, interaction, and price features
│   │   └── lgbm_ranker.py       # LightGBM GBDT Re-ranking model
│   └── evaluation/
│       ├── metrics.py           # MAP@12, Precision@12, Recall@12, HitRate@12
│       └── cold_start.py        # Activity cohort & cold-start segmentation
├── scripts/
│   └── run_pipeline.py          # End-to-end execution and benchmark reporting
├── tests/
│   ├── test_retrieval.py        # Unit tests for Stage 1 candidate generation
│   ├── test_ranking.py          # Unit tests for FeatureBuilder & LightGBM ranker
│   └── test_metrics.py          # Unit tests for MAP@12 and AP calculations
├── pytest.ini
└── README.md
```

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn lightgbm pyyaml pytest pyarrow
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run End-to-End Pipeline
```bash
python scripts/run_pipeline.py
```

---

## 📄 LaTeX Resume Entry

```latex
\entry{Two-Stage Fashion Recommender System}{Python, PyTorch, FAISS, LightGBM}{2026}{
  \item Built candidate retrieval (item/user embeddings + popularity/co-purchase) and LightGBM re-ranker on the H\&M transactions dataset (5,000 customers, 1,000 items).
  \item Improved MAP@12 from 0.0038 to 0.0108 (+184\% relative lift) vs.\ popularity baseline; analysed cold-start handling for low-activity users. \href{https://github.com/Abhinesh-sekar/fashion-recommender-system}{[Code]}
}
```
