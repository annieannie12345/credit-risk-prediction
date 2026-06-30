import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Credit Risk Prediction Engine",
    layout="wide"
)

st.title("Credit Risk Prediction Engine")

st.subheader("Applicant Details")

JOB_LABELS = {
    0: "Unskilled / non-resident",
    1: "Unskilled resident",
    2: "Skilled employee / official",
    3: "Highly skilled / management",
}

HOUSING_LABELS = {
    "own": "Own home",
    "rent": "Rented home",
    "free": "Free housing",
}

ACCOUNT_LABELS = {
    "little": "Low balance",
    "moderate": "Moderate balance",
    "quite rich": "High balance",
    "rich": "Very high balance",
    "NA": "No account / unknown",
}

PURPOSE_LABELS = {
    "radio/TV": "Electronics / radio / TV",
    "education": "Education",
    "furniture/equipment": "Furniture or equipment",
    "car": "Car",
    "business": "Business",
    "domestic appliances": "Home appliances",
    "repairs": "Repairs",
    "vacation/others": "Vacation or other",
}

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Applicant age",
        min_value=18,
        max_value=100,
        value=30,
        help="Age of the person applying for the loan.",
    )
    sex = st.selectbox(
        "Applicant gender",
        ["male", "female"],
        format_func=lambda value: value.title(),
    )
    job = st.selectbox(
        "Employment skill level",
        [0, 1, 2, 3],
        index=2,
        format_func=lambda value: JOB_LABELS[value],
        help="German Credit dataset job category. Higher values usually mean higher skill or responsibility.",
    )
    housing = st.selectbox(
        "Housing status",
        ["own", "rent", "free"],
        format_func=lambda value: HOUSING_LABELS[value],
    )
    saving_accounts = st.selectbox(
        "Savings account balance",
        ["little", "moderate", "quite rich", "rich", "NA"],
        format_func=lambda value: ACCOUNT_LABELS[value],
        help="Approximate savings account balance category.",
    )

with col2:
    checking_account = st.selectbox(
        "Checking account balance",
        ["little", "moderate", "rich", "NA"],
        index=3,
        format_func=lambda value: ACCOUNT_LABELS[value],
        help="Approximate checking account balance category.",
    )
    credit_amount = st.number_input(
        "Requested loan amount",
        min_value=0,
        value=3000,
        step=100,
        help="Total amount of credit requested by the applicant.",
    )
    duration = st.number_input(
        "Loan duration in months",
        min_value=1,
        max_value=100,
        value=24,
        help="Number of months over which the loan will be repaid.",
    )
    purpose = st.selectbox(
        "Loan purpose",
        [
            "radio/TV",
            "education",
            "furniture/equipment",
            "car",
            "business",
            "domestic appliances",
            "repairs",
            "vacation/others"
        ],
        format_func=lambda value: PURPOSE_LABELS[value],
    )

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
    friendly_input_data = pd.DataFrame({
        "Applicant age": [age],
        "Applicant gender": [sex.title()],
        "Employment skill level": [JOB_LABELS[job]],
        "Housing status": [HOUSING_LABELS[housing]],
        "Savings account balance": [ACCOUNT_LABELS[saving_accounts]],
        "Checking account balance": [ACCOUNT_LABELS[checking_account]],
        "Requested loan amount": [credit_amount],
        "Loan duration in months": [duration],
        "Loan purpose": [PURPOSE_LABELS[purpose]],
    })
    st.dataframe(friendly_input_data, use_container_width=True)
