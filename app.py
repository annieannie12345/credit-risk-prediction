"""Interactive credit-risk scoring dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from credit_risk.data import load_credit_data
from credit_risk.explain import local_explanation
from credit_risk.predict import load_bundle, predict_applicant

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"
USD_TO_INR = 94.54


def format_inr(value: int | float) -> str:
    """Format a whole monetary value using Indian lakh/crore grouping."""
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        grouped = digits
    else:
        tail = digits[-3:]
        head = digits[:-3]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        grouped = ",".join([*groups, tail])
    return f"{sign}₹{grouped}"


def inr_to_model_currency(value: int | float) -> float:
    """Convert INR UI inputs into the USD-scale values used by the trained dataset."""
    return float(value) / USD_TO_INR


def add_inr_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add INR display fields without changing the model's original USD-scale data."""
    display_frame = frame.copy()
    display_frame["person_income_inr"] = display_frame["person_income"] * USD_TO_INR
    display_frame["loan_amnt_inr"] = display_frame["loan_amnt"] * USD_TO_INR
    return display_frame


st.set_page_config(
    page_title="Credit Risk Prediction Engine",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_bundle(model_modified_time: float):
    """Reload automatically whenever training replaces the model artifact."""
    return load_bundle(MODEL_PATH)


@st.cache_data
def get_data(data_modified_time: float):
    """Reload automatically whenever the source CSV is replaced."""
    return load_credit_data(DATA_PATH)


bundle = get_bundle(MODEL_PATH.stat().st_mtime)
data = get_data(DATA_PATH.stat().st_mtime)

st.title("Credit Risk Prediction Engine")

score_tab, performance_tab, explain_tab, data_tab = st.tabs(
    ["Applicant Scorer", "Model Performance", "Explainability", "Data Explorer"]
)

with score_tab:
    st.subheader("Enter applicant details")
    left, middle, right = st.columns(3)
    with left:
        person_age = st.slider("Age", 18, 100, 30)
        person_income = st.number_input(
            "Annual income (₹)",
            min_value=50_000,
            max_value=20_000_000,
            value=5_50_000,
            step=10_000,
        )
        st.caption(f"Entered income: {format_inr(person_income)}")
        person_emp_length = st.number_input(
            "Employment length (years)",
            min_value=0.0,
            max_value=float(min(60, person_age - 14)),
            value=4.0,
            step=1.0,
        )
    with middle:
        person_home_ownership = st.selectbox(
            "Home ownership", sorted(data["person_home_ownership"].unique())
        )
        loan_intent = st.selectbox("Loan intent", sorted(data["loan_intent"].unique()))
        loan_grade = st.selectbox("Loan grade", sorted(data["loan_grade"].unique()))
        prior_default = st.selectbox(
            "Previous default on file",
            sorted(data["cb_person_default_on_file"].unique()),
            format_func=lambda value: "Yes" if value == "Y" else "No",
        )
    with right:
        loan_amnt = st.number_input(
            "Loan amount (₹)",
            min_value=10_000,
            max_value=5_000_000,
            value=5_00_000,
            step=10_000,
        )
        st.caption(f"Entered loan: {format_inr(loan_amnt)}")
        loan_int_rate = st.number_input(
            "Interest rate (%)", min_value=0.0, max_value=40.0, value=11.0, step=0.1
        )
        cred_hist_length = st.slider("Credit history length (years)", 1, 50, 6)

    model_person_income = inr_to_model_currency(person_income)
    model_loan_amnt = inr_to_model_currency(loan_amnt)
    applicant = {
        "person_age": person_age,
        "person_income": model_person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": model_loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": model_loan_amnt / max(model_person_income, 1),
        "cb_person_default_on_file": prior_default,
        "cb_person_cred_hist_length": cred_hist_length,
    }

    if st.button("Calculate risk", type="primary", use_container_width=True):
        result = predict_applicant(bundle, applicant)
        probability = result["default_probability"]
        metric_col, action_col = st.columns([1, 2])
        with metric_col:
            st.metric("Probability of default", f"{probability:.1%}")
            st.progress(probability)
        with action_col:
            if result["prediction"] == "High risk":
                st.error(
                    f"High risk — {result['recommended_action']}. "
                    f"The operating threshold is {result['threshold']:.0%}."
                )
            else:
                st.success(
                    f"Lower risk — {result['recommended_action']}. "
                    f"The operating threshold is {result['threshold']:.0%}."
                )

        st.subheader("What drove this score?")
        explanation = local_explanation(bundle, pd.DataFrame([applicant]))
        fig = px.bar(
            explanation.sort_values("impact"),
            x="impact",
            y="feature",
            color="direction",
            orientation="h",
            color_discrete_map={
                "Increases predicted risk": "#DC2626",
                "Decreases predicted risk": "#16A34A",
            },
            labels={"impact": "SHAP impact on default-risk score", "feature": ""},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

with performance_tab:
    st.subheader("Model comparison")
    comparison = pd.read_csv(REPORT_PATH)
    display = comparison[
        [
            "model",
            "selected",
            "test_roc_auc",
            "test_pr_auc",
            "test_precision_bad",
            "test_recall_bad",
            "test_f1_bad",
        ]
    ].copy()
    display.columns = [
        "Model",
        "Deployed",
        "ROC-AUC",
        "PR-AUC",
        "Precision (default)",
        "Recall (default)",
        "F1 (default)",
    ]
    st.dataframe(
        display.style.format(
            {
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Precision (default)": "{:.3f}",
                "Recall (default)": "{:.3f}",
                "F1 (default)": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.image(str(ROOT / "reports" / "figures" / "roc_curves.png"))
    st.info(
        f"{bundle['model_name']} is the deployed model. It achieved the lowest "
        "cross-validated business cost among the compared candidates. "
        "Thresholds use 5-fold out-of-fold predictions on training data. "
        "A missed bad borrower costs 5 units; an unnecessarily rejected good borrower costs 1."
    )

with explain_tab:
    st.subheader(f"Global SHAP explanation — {bundle['model_name']}")
    st.image(str(ROOT / "reports" / "figures" / "shap_summary.png"))
    st.caption(
        "Red points are high feature values and blue points are low values. "
        "Position to the right increases predicted default risk."
    )
    importance = pd.read_csv(ROOT / "reports" / "shap_feature_importance.csv").head(15)
    st.dataframe(importance, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("Dataset health and patterns")
    a, b, c = st.columns(3)
    a.metric("Applicants", f"{len(data):,}")
    b.metric("Default rate", f"{data['loan_status'].mean():.1%}")
    c.metric("Duplicate rows", f"{data.duplicated().sum():,}")

    left, right = st.columns(2)
    with left:
        risk_counts = (
            data["loan_status"]
            .map({0: "Non-default", 1: "Default"})
            .value_counts()
            .rename_axis("risk")
            .reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(risk_counts, x="risk", y="count", color="risk", title="Class balance"),
            use_container_width=True,
        )
    with right:
        loan_plot = add_inr_display_columns(data).assign(
            outcome=data["loan_status"].map({0: "Non-default", 1: "Default"})
        )
        st.plotly_chart(
            px.histogram(
                loan_plot,
                x="loan_amnt_inr",
                color="outcome",
                barmode="overlay",
                nbins=35,
                title="Loan amount (₹) by outcome",
                labels={"loan_amnt_inr": "Loan amount (₹)", "outcome": "Outcome"},
            ),
            use_container_width=True,
        )
