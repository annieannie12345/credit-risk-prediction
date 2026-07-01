"""Stable prediction API used by the Streamlit app and tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from credit_risk.data import RAW_FEATURES


def load_bundle(path: str | Path = "artifacts/best_model.joblib") -> dict:
    return joblib.load(path)


def predict_applicant(bundle: dict, applicant: dict | pd.DataFrame) -> dict:
    frame = pd.DataFrame([applicant]) if isinstance(applicant, dict) else applicant.copy()
    missing = set(RAW_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Applicant is missing fields: {sorted(missing)}")
    frame = frame[RAW_FEATURES]
    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    is_high_risk = probability >= threshold
    return {
        "default_probability": probability,
        "bad_credit_probability": probability,
        "threshold": threshold,
        "prediction": "High risk" if is_high_risk else "Lower risk",
        "recommended_action": "Refer / decline" if is_high_risk else "Consider approval",
    }
