# Credit Risk Prediction Engine

A machine-learning web app that predicts whether a loan applicant is likely to
default. The project compares multiple models, deploys the best-performing
Random Forest model, explains predictions with SHAP, and presents everything in
an easy-to-use Streamlit dashboard.

![Applicant scoring dashboard](assets/1.png)

![Model performance dashboard](assets/2.png)

## What this project does

- Predicts loan default risk from applicant, income, loan, and credit-history
  details.
- Compares Logistic Regression, Random Forest, XGBoost, and CatBoost.
- Uses Random Forest as the deployed model because it gives the strongest
  overall performance in the current evaluation.
- Shows model performance with ROC-AUC, PR-AUC, precision, recall, and F1-score.
- Explains predictions using SHAP so the user can understand which factors
  pushed the risk score higher or lower.
- Converts user-facing monetary values to Indian rupees in the Streamlit app.

> This is an educational decision-support project, not a production lending
> system. A real lending system would also need policy validation, fairness
> testing, legal review, monitoring, and governance.

## Dataset

The app uses a public credit-risk style dataset with applicant and loan fields
such as:

- applicant age and income
- employment length
- home ownership
- loan intent
- loan grade
- loan amount and interest rate
- previous default status
- credit history length
- loan status target

The target column is `loan_status`:

- `0` = non-default
- `1` = default / high-risk loan

## Model performance

Random Forest is currently deployed.

| Metric | Random Forest result |
|---|---:|
| Accuracy | 0.864 |
| ROC-AUC | 0.937 |
| PR-AUC | 0.893 |
| F1-score for default class | 0.734 |
| Operating threshold | 0.29 |

The threshold is optimized for credit-risk decision support instead of simply
using the default `0.5` cutoff.

## How the project works

1. Load and validate the dataset.
2. Clean duplicate and impossible records.
3. Engineer useful features such as monthly income, interest proxy, age bands,
   and income bands.
4. Preprocess numerical and categorical columns safely inside a scikit-learn
   pipeline.
5. Train and compare multiple machine-learning models.
6. Select the deployed model and threshold.
7. Generate model reports, ROC curves, confusion matrices, and SHAP outputs.
8. Serve predictions through the Streamlit dashboard.

## Project structure

```text
credit_risk_prediction_engine/
├── app.py                         # Streamlit dashboard
├── assets/                        # README screenshots
│   ├── 1.png
│   └── 2.png
├── data/
│   └── credit_risk_dataset.csv    # Source dataset
├── reports/                       # Evaluation reports and plots
├── src/credit_risk/
│   ├── data.py                    # Data loading and validation
│   ├── features.py                # Feature engineering
│   ├── modeling.py                # Models, metrics, threshold selection
│   ├── explain.py                 # SHAP explainability
│   ├── predict.py                 # Prediction helper
│   └── train.py                   # Training script
├── tests/
│   └── test_core.py               # Basic automated tests
├── requirements.txt
└── pyproject.toml
```

## How to run locally

Clone the repository and enter the project folder:

```bash
git clone git@github.com:annieannie12345/credit-risk-prediction.git
cd credit-risk-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train or refresh the model reports:

```bash
python -m credit_risk.train --model "Random Forest"
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How to use the app

1. Open the **Applicant Scorer** tab.
2. Enter applicant details such as age, income, home ownership, loan amount,
   loan intent, loan grade, interest rate, previous default status, and credit
   history length.
3. Click **Calculate risk**.
4. Review the predicted default probability and recommendation.
5. Check the SHAP explanation to understand the main factors affecting the
   prediction.

## Run tests

```bash
pytest -q
```

Expected result:

```text
3 passed
```

## Key learning points

This project demonstrates:

- end-to-end machine-learning project structure
- model comparison and evaluation
- threshold tuning for business-sensitive classification
- Streamlit dashboard development
- SHAP-based model explainability
- reproducible training and testing workflow

## Interview-ready summary

I built an end-to-end credit-risk prediction system using a 32K-row loan dataset.
The project compares Logistic Regression, Random Forest, XGBoost, and CatBoost,
deploys Random Forest based on the strongest evaluation results, tunes the
classification threshold for lending risk, explains predictions with SHAP, and
serves the full workflow through a Streamlit dashboard.
