"""Feature engineering used identically during training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "monthly_income",
    "interest_amount_proxy",
    "employment_age_ratio",
    "credit_history_age_ratio",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "age_band",
    "income_band",
]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add transparent lending features without using the target."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["monthly_income"] = frame["person_income"] / 12
        frame["interest_amount_proxy"] = (
            frame["loan_amnt"] * frame["loan_int_rate"] / 100
        )
        frame["employment_age_ratio"] = (
            frame["person_emp_length"] / frame["person_age"].clip(lower=1)
        )
        frame["credit_history_age_ratio"] = (
            frame["cb_person_cred_hist_length"] / frame["person_age"].clip(lower=1)
        )
        frame["age_band"] = pd.cut(
            frame["person_age"],
            bins=[0, 25, 35, 50, np.inf],
            labels=["under_25", "25_to_34", "35_to_49", "50_plus"],
            right=False,
        ).astype(object)
        frame["income_band"] = pd.cut(
            frame["person_income"],
            bins=[0, 30_000, 60_000, 100_000, np.inf],
            labels=["under_30k", "30k_to_59k", "60k_to_99k", "100k_plus"],
            right=False,
        ).astype(object)
        return frame
