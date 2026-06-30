import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import load_data, preprocess_data
from evaluate import evaluate_model


DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "models/best_model.pkl"


def main():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df,
        target_column="Risk"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0,
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        roc_auc = evaluate_model(name, pipeline, X_test, y_test)

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\nBest Model:", best_model_name)
    print("Best ROC-AUC:", round(best_score, 4))
    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
