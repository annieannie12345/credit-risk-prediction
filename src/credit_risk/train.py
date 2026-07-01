"""Train every candidate and save the selected explainable model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from credit_risk.data import dataset_profile, load_credit_data, split_xy
from credit_risk.explain import compute_shap_values, friendly_feature_name, transformed_data
from credit_risk.modeling import select_best_model, train_and_evaluate


def save_evaluation_plots(results, y_test, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for result in results:
        RocCurveDisplay.from_predictions(
            y_test,
            result.test_probabilities,
            name=f"{result.name} (AUC={result.test_metrics['roc_auc']:.3f})",
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    ax.set_title("Held-out ROC curves")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    for result in results:
        predictions = (result.test_probabilities >= result.threshold).astype(int)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["No default", "Default"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"{result.name} at threshold {result.threshold:.2f}")
        fig.tight_layout()
        slug = result.name.lower().replace(" ", "_")
        fig.savefig(figures_dir / f"confusion_matrix_{slug}.png", dpi=160)
        plt.close(fig)


def save_shap_outputs(bundle: dict, sample: pd.DataFrame, reports_dir: Path) -> None:
    shap_values, names = compute_shap_values(bundle, sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": [friendly_feature_name(name) for name in names],
            "mean_absolute_shap": mean_abs,
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(reports_dir / "shap_feature_importance.csv", index=False)

    transformed, _ = transformed_data(bundle, sample)
    explanation = shap.Explanation(
        values=shap_values,
        data=transformed,
        feature_names=[friendly_feature_name(name) for name in names],
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"SHAP summary — {bundle['model_name']}")
    plt.tight_layout()
    plt.savefig(reports_dir / "figures" / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def run_training(
    data_path: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    deployed_model: str = "Random Forest",
) -> dict:
    np.random.seed(42)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_credit_data(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    recommended = select_best_model(results)
    matches = [result for result in results if result.name == deployed_model]
    if not matches:
        available = ", ".join(result.name for result in results)
        raise ValueError(f"Unknown deployed model '{deployed_model}'. Choose from: {available}")
    selected = matches[0]

    comparison_rows = []
    for result in results:
        comparison_rows.append(
            {
                "model": result.name,
                **{f"cv_{k}": v for k, v in result.cv_metrics.items()},
                **{f"test_{k}": v for k, v in result.test_metrics.items()},
                "selected": result.name == selected.name,
                "cost_recommended": result.name == recommended.name,
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_business_cost", "cv_roc_auc"], ascending=[True, False]
    )
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    test_predictions = X_test.copy()
    test_predictions["actual_status"] = y_test.map({0: "non_default", 1: "default"})
    for result in results:
        slug = result.name.lower().replace(" ", "_")
        test_predictions[f"{slug}_default_probability"] = result.test_probabilities
        test_predictions[f"{slug}_prediction"] = np.where(
            result.test_probabilities >= result.threshold, "default", "non_default"
        )
    test_predictions.to_csv(reports_dir / "test_predictions.csv", index=False)

    background, _ = transformed_data(
        {"pipeline": selected.pipeline},
        X_train.sample(n=min(100, len(X_train)), random_state=42),
    )
    profile = dataset_profile(df)
    bundle = {
        "pipeline": selected.pipeline,
        "model_name": selected.name,
        "threshold": selected.threshold,
        "cv_metrics": selected.cv_metrics,
        "test_metrics": selected.test_metrics,
        "feature_columns": X.columns.tolist(),
        "explanation_background": background,
        "dataset_profile": profile,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "business_costs": {"false_approval": 5, "false_rejection": 1},
        "selection_policy": "Explicitly deployed model",
        "cost_recommended_model": recommended.name,
    }
    joblib.dump(bundle, artifacts_dir / "best_model.joblib")

    save_evaluation_plots(results, y_test, reports_dir / "figures")
    shap_sample = X_test.sample(n=min(150, len(X_test)), random_state=42)
    save_shap_outputs(bundle, shap_sample, reports_dir)

    summary = {
        "selected_model": selected.name,
        "cost_recommended_model": recommended.name,
        "threshold": selected.threshold,
        "cv_metrics": selected.cv_metrics,
        "test_metrics": selected.test_metrics,
        "dataset_profile": profile,
    }
    (reports_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/credit_risk_dataset.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--model",
        default="Random Forest",
        choices=["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"],
        help="Model to deploy after comparing all candidates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_training(
        args.data,
        args.artifacts_dir,
        args.reports_dir,
        deployed_model=args.model,
    )
    print(json.dumps(result, indent=2))
