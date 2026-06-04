"""Generate a realistic synthetic Telco-style churn dataset (no download needed).

Churn is driven by interpretable signals (short tenure, month-to-month contract,
high monthly charges, fiber + no tech support) plus noise — so models find real,
explainable structure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "churn.csv"


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    tenure = rng.integers(0, 73, n)
    monthly = np.where(internet == "Fiber optic", rng.normal(85, 15, n),
              np.where(internet == "DSL", rng.normal(55, 12, n), rng.normal(20, 5, n))).clip(15, 130).round(2)
    tech_support = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)

    # latent churn propensity
    z = (-1.6
         + 1.5 * (contract == "Month-to-month")
         - 1.2 * (contract == "Two year")
         + 0.9 * (internet == "Fiber optic")
         - 0.8 * (tech_support == "Yes")
         + 0.02 * (monthly - 60)
         - 0.04 * tenure
         + 0.4 * senior
         + 0.3 * (payment == "Electronic check")
         + rng.normal(0, 0.5, n))
    churn = (1 / (1 + np.exp(-z)) > rng.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "senior_citizen": senior,
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "tech_support": tech_support,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": (monthly * np.maximum(tenure, 1)).round(2),
        "churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}  | churn rate = {df.churn.mean():.1%}")
