"""Model construction, threshold selection, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    CreditFeatureEngineer,
)


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    threshold: float
    cv_metrics: dict
    test_metrics: dict
    test_probabilities: np.ndarray


def build_preprocessor() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", CreditFeatureEngineer()),
            ("columns", columns),
        ]
    )


def candidate_models(random_state: int = 42) -> dict:
    """Return deliberately compact, reproducible model candidates."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            C=0.7,
            solver="liblinear",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=2.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=random_state,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    false_approval_cost: float = 5.0,
    false_rejection_cost: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """Minimize lending cost: a bad borrower approved is a false negative."""
    rows = []
    for threshold in np.round(np.arange(0.05, 0.81, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        cost = false_approval_cost * fn + false_rejection_cost * fp
        rows.append(
            {
                "threshold": threshold,
                "business_cost": float(cost),
                "recall_bad": recall_score(y_true, predictions, zero_division=0),
                "precision_bad": precision_score(y_true, predictions, zero_division=0),
                "false_approvals": int(fn),
                "false_rejections": int(fp),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["business_cost", "recall_bad", "precision_bad"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"]), table


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_bad": precision_score(y_true, predictions, zero_division=0),
        "recall_bad": recall_score(y_true, predictions, zero_division=0),
        "f1_bad": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "specificity_good": tn / (tn + fp) if tn + fp else 0.0,
        "false_approvals": int(fn),
        "false_rejections": int(fp),
        "business_cost": float(5 * fn + fp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> list[ModelResult]:
    """Compare candidates using training-only OOF predictions, then test once."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []

    for name, estimator in candidate_models(random_state).items():
        pipeline = make_pipeline(estimator)
        oof_probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold, _ = select_threshold(y_train, oof_probabilities)
        cv_metrics = calculate_metrics(y_train, oof_probabilities, threshold)

        pipeline.fit(X_train, y_train)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, test_probabilities, threshold)
        results.append(
            ModelResult(
                name=name,
                pipeline=pipeline,
                threshold=threshold,
                cv_metrics=cv_metrics,
                test_metrics=test_metrics,
                test_probabilities=test_probabilities,
            )
        )
    return results


def select_best_model(results: list[ModelResult]) -> ModelResult:
    """Select on cross-validated business cost, breaking ties with discrimination."""
    return sorted(
        results,
        key=lambda result: (
            result.cv_metrics["business_cost"],
            -result.cv_metrics["roc_auc"],
            -result.cv_metrics["recall_bad"],
        ),
    )[0]
