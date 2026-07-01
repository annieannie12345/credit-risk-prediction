"""Dataset loading, validation, cleaning, and train/test preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

TARGET = "loan_status"


def load_credit_data(path: str | Path) -> pd.DataFrame:
    """Load, validate, and clean the credit-risk dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = set(RAW_FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[RAW_FEATURES + [TARGET]].copy()
    for column in [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        TARGET,
    ]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    unexpected_targets = set(df[TARGET].dropna().unique()).difference({0, 1})
    if unexpected_targets or df[TARGET].isna().any():
        raise ValueError(f"Unexpected target values: {sorted(unexpected_targets)}")

    # Remove exact duplicates and impossible source-data anomalies.
    df = df.drop_duplicates()
    valid_age = df["person_age"].between(18, 100)
    valid_income = df["person_income"] > 0
    valid_loan = df["loan_amnt"] > 0
    valid_loan_ratio = df["loan_percent_income"].between(0, 1)
    plausible_employment = (
        df["person_emp_length"].isna()
        | (
            (df["person_emp_length"] >= 0)
            & (df["person_emp_length"] <= 60)
            & (df["person_emp_length"] <= df["person_age"] - 14)
        )
    )
    df = df.loc[
        valid_age & valid_income & valid_loan & valid_loan_ratio & plausible_employment
    ]
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw predictors and loan status where risky/defaulted is 1."""
    return df[RAW_FEATURES].copy(), df[TARGET].astype(int)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Create compact data-quality metadata for reports and the UI."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "default_rate": float(df[TARGET].mean()),
        "duplicate_rows_after_cleaning": int(df.duplicated().sum()),
        "missing_by_column": {
            column: int(count) for column, count in df.isna().sum().items() if count
        },
    }
