"""
evaluate.py
-----------
Evaluate trained models, save metrics to results/model_metrics.csv,
and generate report figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ── Metrics ──────────────────────────────────────────────────────────────────

def build_results_table(models: dict, X_test, y_test, times: dict) -> pd.DataFrame:
    """
    Build a results DataFrame for all models.

    Parameters
    ----------
    models : dict  {name: fitted_model}
    X_test : array-like
    y_test : array-like
    times  : dict  {name: training_time_seconds}
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        row = {
            "Model": name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            "Training Time (s)": times.get(name),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_metrics(results: pd.DataFrame, output_path: str = "results/model_metrics.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Metrics saved → {output_path}")
    print(results.round(4).to_string(index=False))


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_model_comparison(results: pd.DataFrame,
                           output_path: str = "reports/figures/model_comparison.png"):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_melt = results.melt(id_vars="Model", value_vars=metrics,
                            var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", ax=ax, palette="viridis")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {output_path}")


def plot_confusion_matrices(models: dict, X_test, y_test,
                             output_path: str = "reports/figures/confusion_matrices.png"):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {output_path}")


def plot_feature_importance(models: dict, feature_names: list,
                             top_n: int = 15,
                             output_path: str = "reports/figures/feature_importance.png"):
    """Plot feature importances for tree-based models."""
    tree_models = {
        name: m for name, m in models.items()
        if hasattr(m, "feature_importances_")
    }
    if not tree_models:
        print("No tree-based models found; skipping feature importance plot.")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances.nlargest(top_n).sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"{name} — Top {top_n} Features", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {output_path}")
