"""SHAP utilities for global and applicant-level explanations."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import shap


def transformed_data(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    values = preprocessor.transform(X)
    names = preprocessor.named_steps["columns"].get_feature_names_out().tolist()
    return np.asarray(values), names


def _positive_class_values(explanation) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def build_explainer(bundle: dict):
    """Build a model-appropriate SHAP explainer from stored background data."""
    background = np.asarray(bundle["explanation_background"])
    model = bundle["pipeline"].named_steps["model"]
    name = bundle["model_name"]
    if name == "Logistic Regression":
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model, background)


def compute_shap_values(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    values, names = transformed_data(bundle, X)
    explainer = build_explainer(bundle)
    explanation = explainer(values, check_additivity=False)
    return _positive_class_values(explanation), names


def friendly_feature_name(transformed_name: str) -> str:
    clean = re.sub(r"^(num|cat)__", "", transformed_name)
    clean = clean.replace("_", " ")
    return clean.title()


def local_explanation(bundle: dict, applicant: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    shap_values, names = compute_shap_values(bundle, applicant)
    row = shap_values[0]
    result = pd.DataFrame(
        {
            "feature": [friendly_feature_name(name) for name in names],
            "impact": row,
        }
    )
    result["direction"] = np.where(
        result["impact"] >= 0,
        "Increases predicted risk",
        "Decreases predicted risk",
    )
    result["absolute_impact"] = result["impact"].abs()
    return result.nlargest(top_n, "absolute_impact").drop(columns="absolute_impact")

