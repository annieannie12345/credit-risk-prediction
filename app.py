import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.pkl"
EUR_TO_INR = 107.8565

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Credit Risk Prediction Engine",
    layout="wide"
)

st.title("Credit Risk Prediction Engine")

st.subheader("Applicant Details")

JOB_OPTIONS = {
    "0 - Unskilled / non-resident": 0,
    "1 - Unskilled resident": 1,
    "2 - Skilled employee / official": 2,
    "3 - Highly skilled / management": 3,
}

HOUSING_OPTIONS = {
    "Own home": "own",
    "Rented home": "rent",
    "Free housing": "free",
}

ACCOUNT_OPTIONS = {
    "Low balance": "little",
    "Moderate balance": "moderate",
    "High balance": "quite rich",
    "Very high balance": "rich",
    "No account / unknown": np.nan,
}

CHECKING_ACCOUNT_OPTIONS = {
    "Low balance": "little",
    "Moderate balance": "moderate",
    "High balance": "rich",
    "No account / unknown": np.nan,
}

PURPOSE_OPTIONS = {
    "Electronics / radio / TV": "radio/TV",
    "Education": "education",
    "Furniture or equipment": "furniture/equipment",
    "Car": "car",
    "Business": "business",
    "Home appliances": "domestic appliances",
    "Repairs": "repairs",
    "Vacation or other": "vacation/others",
}

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Applicant Age", min_value=19, max_value=75, value=30)
    sex_label = st.selectbox("Gender", ["Male", "Female"])
    job_label = st.selectbox(
        "Job Type",
        list(JOB_OPTIONS.keys()),
        index=2
    )
    housing_label = st.selectbox("Housing Status", list(HOUSING_OPTIONS.keys()))
    saving_accounts_label = st.selectbox(
        "Savings Account Balance",
        list(ACCOUNT_OPTIONS.keys())
    )

with col2:
    checking_account_label = st.selectbox(
        "Checking Account Balance",
        list(CHECKING_ACCOUNT_OPTIONS.keys())
    )
    loan_amount_inr = st.number_input(
        "Loan Amount (INR)",
        min_value=round(250 * EUR_TO_INR),
        max_value=round(18424 * EUR_TO_INR),
        value=round(3000 * EUR_TO_INR),
        step=1000
    )
    duration = st.number_input(
        "Loan Duration",
        min_value=4,
        max_value=72,
        value=24
    )
    purpose_label = st.selectbox("Loan Purpose", list(PURPOSE_OPTIONS.keys()))

sex = sex_label.lower()
job = JOB_OPTIONS[job_label]
housing = HOUSING_OPTIONS[housing_label]
saving_accounts = ACCOUNT_OPTIONS[saving_accounts_label]
checking_account = CHECKING_ACCOUNT_OPTIONS[checking_account_label]
credit_amount = loan_amount_inr / EUR_TO_INR
purpose = PURPOSE_OPTIONS[purpose_label]

input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Job": [job],
    "Housing": [housing],
    "Saving accounts": [saving_accounts],
    "Checking account": [checking_account],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Purpose": [purpose]
})

st.divider()

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk Applicant")
    else:
        st.success("Low Risk Applicant")

    st.metric("Default Probability", f"{probability:.2%}")

    if probability >= 0.70:
        st.warning("Business Decision: Reject Loan Application")
    elif probability >= 0.40:
        st.warning("Business Decision: Send For Manual Review")
    else:
        st.success("Business Decision: Approve Loan Application")

    st.subheader("Input Summary")
    display_data = pd.DataFrame({
        "Applicant Age": [age],
        "Gender": [sex_label],
        "Job Type": [job_label],
        "Housing Status": [housing_label],
        "Savings Account Balance": [saving_accounts_label],
        "Checking Account Balance": [checking_account_label],
        "Loan Amount": [f"₹{loan_amount_inr:,.0f}"],
        "Loan Duration": [f"{duration} months"],
        "Loan Purpose": [purpose_label],
    })
    st.dataframe(display_data, use_container_width=True)
