# Credit Risk Prediction Engine

Machine learning project that predicts loan default risk using the German Credit dataset.

## Features

- Data preprocessing for numeric and categorical applicant features
- Model training with Logistic Regression, Random Forest, XGBoost, and CatBoost
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Saved best model for reuse
- Streamlit dashboard for real-time credit risk prediction

## How To Run

```bash
pip install -r requirements.txt
python3 src/train_model.py
streamlit run app.py
```

## Dataset

The app expects the dataset at:

```text
data/loan_data.csv
```
