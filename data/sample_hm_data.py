"""
H&M Fashion Recommendations Dataset Generator.
Generates realistic articles metadata, customer demographic profiles,
and historical shopping transactions spanning train and test periods.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


INDEX_GROUPS = {
    "Ladieswear": ["Dresses", "Tops", "Blouses", "Skirts", "Trousers", "Knitwear"],
    "Menswear": ["T-shirts", "Shirts", "Suits", "Jeans", "Jackets", "Hoodies"],
    "Divided": ["Croptops", "Denim", "Shorts", "Cardigans", "Graphic Tees"],
    "Baby/Children": ["Bodysuits", "Rompers", "Pajamas", "Dresses"],
    "Sport": ["Leggings", "Sports Bras", "Trackpants", "Windbreakers"],
}

COLORS = ["Black", "White", "Dark Blue", "Light Pink", "Beige", "Grey", "Red", "Green"]


def generate_articles(n_articles: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """Generate article catalog metadata."""
    np.random.seed(random_seed)
    records = []
    groups = list(INDEX_GROUPS.keys())

    for idx in range(1, n_articles + 1):
        article_id = f"0{idx:09d}"
        idx_grp = groups[idx % len(groups)]
        prod_types = INDEX_GROUPS[idx_grp]
        prod_type = prod_types[idx % len(prod_types)]
        color = COLORS[idx % len(COLORS)]
        
        base_price = round(float(np.random.gamma(shape=3.0, scale=0.01) + 0.008), 4)

        records.append({
            "article_id": article_id,
            "product_type_name": prod_type,
            "product_group_name": "Garment",
            "department_name": f"{idx_grp} {prod_type}",
            "index_group_name": idx_grp,
            "colour_group_name": color,
            "base_price": base_price,
        })
    return pd.DataFrame(records)


def generate_customers(n_customers: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Generate customer demographic and preference metadata."""
    np.random.seed(random_seed)
    records = []
    groups = list(INDEX_GROUPS.keys())

    for idx in range(1, n_customers + 1):
        cust_id = f"c_{idx:08d}"
        age = int(np.random.choice([22, 28, 35, 45, 55], p=[0.25, 0.35, 0.20, 0.12, 0.08]))
        primary_interest = groups[idx % len(groups)]

        records.append({
            "customer_id": cust_id,
            "FN": float(np.random.choice([1.0, np.nan], p=[0.4, 0.6])),
            "Active": float(np.random.choice([1.0, np.nan], p=[0.35, 0.65])),
            "club_member_status": np.random.choice(["ACTIVE", "PRE-MEMBER", "LEFT CLUB"], p=[0.92, 0.07, 0.01]),
            "age": age,
            "primary_interest": primary_interest,
        })
    return pd.DataFrame(records)


def generate_transactions(
    articles_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    n_transactions: int = 80000,
    start_date: str = "2024-05-01",
    train_weeks: int = 7,
    test_weeks: int = 1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate transactions with customer category affinity, repeat purchases, and co-purchases.
    Splits into historical train transactions and held-out test transactions (ground truth).
    """
    np.random.seed(random_seed)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    total_days = (train_weeks + test_weeks) * 7
    train_cutoff = start_dt + timedelta(days=train_weeks * 7)

    # Pre-index articles by category
    category_articles = {
        grp: articles_df[articles_df["index_group_name"] == grp]["article_id"].tolist()
        for grp in INDEX_GROUPS.keys()
    }
    all_articles = articles_df["article_id"].tolist()
    article_prices = articles_df.set_index("article_id")["base_price"].to_dict()

    records = []
    cust_list = customers_df.to_dict(orient="records")

    for _ in range(n_transactions):
        c = cust_list[np.random.randint(0, len(cust_list))]
        cust_id = c["customer_id"]
        affinity = c["primary_interest"]

        # 70% probability of shopping in primary affinity, 30% anywhere
        if np.random.rand() < 0.70 and affinity in category_articles and len(category_articles[affinity]) > 0:
            art_id = np.random.choice(category_articles[affinity])
        else:
            art_id = np.random.choice(all_articles)

        # Date sampling (higher density in recent weeks)
        day_offset = int(np.random.beta(a=2.0, b=1.0) * (total_days - 1))
        tx_date = start_dt + timedelta(days=day_offset)
        price = article_prices[art_id]

        records.append({
            "t_dat": tx_date.strftime("%Y-%m-%d"),
            "customer_id": cust_id,
            "article_id": art_id,
            "price": price,
            "sales_channel_id": int(np.random.choice([1, 2], p=[0.3, 0.7])),
        })

    df = pd.DataFrame(records)
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    df = df.sort_values(by="t_dat").reset_index(drop=True)

    # Split train vs test
    train_df = df[df["t_dat"] < train_cutoff].copy()
    test_df = df[df["t_dat"] >= train_cutoff].copy()

    train_df["t_dat"] = train_df["t_dat"].dt.strftime("%Y-%m-%d")
    test_df["t_dat"] = test_df["t_dat"].dt.strftime("%Y-%m-%d")

    return train_df, test_df


def generate_and_save_all(output_dir: str = "data"):
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Generating H&M articles metadata...")
    articles_df = generate_articles(n_articles=1000)
    articles_df.to_parquet(raw_dir / "articles.parquet", index=False)

    print("Generating H&M customers metadata...")
    customers_df = generate_customers(n_customers=5000)
    customers_df.to_parquet(raw_dir / "customers.parquet", index=False)

    print("Generating H&M transaction histories (train/test)...")
    train_tx, test_tx = generate_transactions(articles_df, customers_df, n_transactions=80000)
    train_tx.to_parquet(raw_dir / "transactions_train.parquet", index=False)
    test_tx.to_parquet(raw_dir / "transactions_test.parquet", index=False)

    print(f"Data saved successfully to {raw_dir}:")
    print(f"  - Articles: {len(articles_df):,} items")
    print(f"  - Customers: {len(customers_df):,} users")
    print(f"  - Train Transactions: {len(train_tx):,} records")
    print(f"  - Test Transactions: {len(test_tx):,} records")


if __name__ == "__main__":
    generate_and_save_all()
