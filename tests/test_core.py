from pathlib import Path

import pandas as pd

from credit_risk.data import RAW_FEATURES, load_credit_data, split_xy
from credit_risk.features import CreditFeatureEngineer
from credit_risk.modeling import select_threshold

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_schema_and_target():
    df = load_credit_data(ROOT / "data" / "credit_risk_dataset.csv")
    X, y = split_xy(df)
    assert len(df) == 32_409
    assert X.columns.tolist() == RAW_FEATURES
    assert set(y.unique()) == {0, 1}
    assert 0.20 < y.mean() < 0.24
    assert df.duplicated().sum() == 0
    assert df["person_age"].max() <= 100


def test_feature_engineering_is_finite():
    sample = pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 60_000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 12_000,
                "loan_int_rate": 10,
                "loan_percent_income": 0.2,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6,
            }
        ]
    )
    result = CreditFeatureEngineer().fit_transform(sample)
    assert result.loc[0, "monthly_income"] == 5_000
    assert result.loc[0, "interest_amount_proxy"] == 1_200


def test_cost_sensitive_threshold_prefers_recall():
    y = pd.Series([0, 0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.6, 0.3, 0.4]).to_numpy()
    threshold, table = select_threshold(y, probabilities)
    assert 0.0 < threshold <= 0.4
    assert not table.empty
