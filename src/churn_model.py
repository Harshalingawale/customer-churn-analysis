"""Reproducible churn EDA + modeling. Saves figures and prints metrics.

    python -m src.churn_model
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "churn.csv"

CATEG = ["contract", "internet_service", "tech_support", "paperless_billing", "payment_method"]
NUM = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]


def load() -> pd.DataFrame:
    if not DATA.exists():
        from data.generate import generate
        generate().to_csv(DATA, index=False)
    return pd.read_csv(DATA)


def eda(df: pd.DataFrame) -> None:
    # Churn by contract type
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    df.groupby("contract")["churn"].mean().sort_values().plot.bar(ax=ax[0], color="#e45756")
    ax[0].set_title("Churn rate by contract"); ax[0].set_ylabel("churn rate")
    for c in ["Yes", "No"]:
        sub = df[df.churn == (1 if c == "Yes" else 0)]
        ax[1].hist(sub.tenure_months, bins=24, alpha=0.6, label=f"churn={c}")
    ax[1].set_title("Tenure distribution by churn"); ax[1].set_xlabel("tenure (months)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "eda_overview.png", dpi=120); plt.close(fig)


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def main() -> dict:
    df = load()
    eda(df)
    X, y = df[NUM + CATEG], df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        pipe = build_pipeline(m).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        results[name] = round(float(roc_auc_score(yte, proba)), 4)
        fitted[name] = (pipe, proba)

    best = max(results, key=results.get)
    pipe, proba = fitted[best]

    # ROC curve
    fpr, tpr, _ = roc_curve(yte, proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"{best} (AUC={results[best]})", color="#4c78a8")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curve"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "roc_curve.png", dpi=120); plt.close(fig)

    # Feature importance (RF)
    rf_pipe = build_pipeline(models["rf"]).fit(Xtr, ytr)
    names = rf_pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=names).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#54a24b"); ax.set_title("Top churn drivers (RF importance)")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=120); plt.close(fig)

    preds = (proba >= 0.5).astype(int)
    report = {
        "auc_by_model": results,
        "best_model": best,
        "confusion_matrix": confusion_matrix(yte, preds).tolist(),
        "classification_report": classification_report(yte, preds, output_dict=True),
        "churn_rate": round(float(df.churn.mean()), 4),
        "top_drivers": list(imp.index[::-1][:5]),
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"auc_by_model": results, "best": best, "top_drivers": report["top_drivers"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
