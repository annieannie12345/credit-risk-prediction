# Credit Risk Prediction Engine

An end-to-end, explainable machine-learning project that predicts the probability
of **loan default** from an applicant's financial and credit-history attributes.
It compares Logistic Regression, Random Forest, XGBoost, and CatBoost, deploys
Random Forest with a lending-specific decision threshold, explains predictions with
SHAP, and serves the result in a Streamlit dashboard.

> This is an educational decision-support project, not a production underwriting
> system. The public credit-risk dataset is historic and contains underwriting
> attributes that require legal, fairness, and governance review before any
> real lending use.

## What was built

```text
credit_risk_prediction_engine/
├── app.py                         # Streamlit UI
├── data/credit_risk_dataset.csv   # Source data
├── src/credit_risk/
│   ├── data.py                    # Loading and schema validation
│   ├── features.py                # Reusable feature engineering
│   ├── modeling.py                # Pipelines, models, metrics, threshold
│   ├── explain.py                 # SHAP explanations
│   ├── predict.py                 # Prediction API
│   └── train.py                   # End-to-end training command
├── tests/test_core.py
├── artifacts/best_model.joblib    # Created by training
└── reports/                       # Metrics, plots, predictions, SHAP output
```

## Run it

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m credit_risk.train
streamlit run app.py
```

Open the local address printed by Streamlit (normally
`http://localhost:8501`). Run tests with:

```bash
pytest -q
```

The dashboard presents monetary inputs and dataset values in Indian rupees.
The Kaggle-style source data uses USD-scale money fields, so the Streamlit app
converts them at `1 USD = ₹94.54`. User-entered INR values are converted back to
the original model scale before prediction, which keeps the trained model
consistent while making the final application India-friendly.

## How the project works, step by step

### 1. Define the business problem

The target is `loan_status`: risky/defaulted loans are 1 and non-defaulted loans
are 0. This convention makes recall mean “what share of truly risky loans did
we catch?”

The two mistakes are not equally expensive:

- **False approval (false negative):** a bad applicant is predicted good — cost 5.
- **False rejection (false positive):** a good applicant is predicted bad — cost 1.

These are demonstration assumptions. A real lender should replace them with
validated expected-loss and opportunity-cost estimates.

### 2. Inspect and clean the data

`data.py` validates the required schema, removes exact duplicates, and excludes
impossible source anomalies such as age over 100 or employment longer than the
applicant's plausible working lifetime. Missing employment length and interest
rate values are retained for pipeline imputation.

### 3. Engineer useful features

`features.py` adds:

- monthly income
- a simple interest-amount proxy
- employment-to-age and credit-history-to-age ratios
- understandable age and income bands

The feature transformer lives inside the model pipeline. That prevents
training/serving skew because Streamlit executes the exact same code as training.

### 4. Preprocess without leakage

Numerical values are median-imputed and standardized. Categorical values are
imputed with `unknown` and one-hot encoded. All preprocessing is fitted only on
the training folds.

### 5. Compare four models fairly

The system evaluates Logistic Regression, Random Forest, XGBoost, and CatBoost.
A stratified 80/20 train/test split preserves the approximately 78/22 class balance.

On the 80% training portion, 5-fold out-of-fold probabilities are generated for
every model. The operating threshold is selected from those training-only
predictions by minimizing the business cost. Random Forest is the configured
production model because it currently gives the strongest overall performance
among the compared candidates. The untouched test set is used once for final
reporting.

### 6. Explain the result

SHAP assigns each feature a signed contribution:

- positive SHAP value → pushes the result toward default;
- negative SHAP value → pushes it toward non-default.

Training saves a global SHAP summary and the dashboard computes local SHAP
drivers for each entered applicant.

### 7. Serve predictions

`app.py` provides four views:

1. applicant scoring and local explanation;
2. model performance;
3. global SHAP explanation;
4. data-quality and distribution exploration.

### 8. Reproduce and extend

Edit model parameters in `candidate_models()`, rerun
`python -m credit_risk.train --model "Random Forest"`, and refresh Streamlit. Useful next steps include
probability calibration, fairness testing across protected groups, temporal
validation, drift monitoring, and an API layer.

## Interview-ready summary

“I built a leakage-safe classifier on a 32K-row credit-risk dataset. I
compared a linear baseline and three tree ensembles using stratified
cross-validation. Because false approvals are more expensive than false
rejections, I optimized the decision threshold using a 5:1 cost ratio instead
of defaulting to 0.5. I evaluated the selected model on a held-out test set,
added global and local SHAP explanations, and packaged the full preprocessing
pipeline into a Streamlit decision-support app.”
