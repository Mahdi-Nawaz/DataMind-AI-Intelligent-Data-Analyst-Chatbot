"""Generate a realistic sample sales dataset for demos. Run: python make_sample_data.py"""
import numpy as np
import pandas as pd


def make_sample(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.2, 0.3, 0.2])
    category = rng.choice(["Electronics", "Furniture", "Clothing", "Grocery"], n, p=[0.25, 0.15, 0.3, 0.3])
    base = {"Electronics": 450, "Furniture": 300, "Clothing": 60, "Grocery": 25}
    unit_price = np.array([base[c] for c in category]) * rng.uniform(0.8, 1.3, n)
    quantity = rng.integers(1, 10, n)
    discount = rng.choice([0, 0.05, 0.1, 0.2], n, p=[0.5, 0.2, 0.2, 0.1])
    seasonal = 1 + 0.25 * np.sin((dates.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi)
    sales = unit_price * quantity * (1 - discount) * seasonal
    profit = sales * rng.normal(0.18, 0.06, n) - discount * sales * 0.3
    df = pd.DataFrame({
        "order_date": dates, "region": region, "category": category,
        "customer_segment": rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[0.55, 0.25, 0.2]),
        "unit_price": unit_price.round(2), "quantity": quantity, "discount": discount,
        "sales": sales.round(2), "profit": profit.round(2),
    })
    # a little realistic mess
    df.loc[rng.choice(n, 40, replace=False), "discount"] = np.nan
    return df.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    import os
    os.makedirs("sample_data", exist_ok=True)
    make_sample().to_csv("sample_data/sales_sample.csv", index=False)
    print("Wrote sample_data/sales_sample.csv")
