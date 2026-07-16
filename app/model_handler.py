import joblib
import mlflow
import pandas as pd
import os


# Global variables for model artifacts
_model = None
_selected_features = None


def load_model():
    global _model, _selected_features
    
    if _model is not None and _selected_features is not None:
        return _model, _selected_features
    
    # Always load from local files inside the container
    _model = joblib.load('models/final_model.pkl')
    _selected_features = joblib.load('models/selected_features.pkl')
    print("Model loaded from local files")
    
    return _model, _selected_features


def predict_single(features: list) -> dict:
    """Predict for a single company."""
    model, selected_features = load_model()
    
    # Create DataFrame with proper column names
    cols = [f'X{i}' for i in range(1, 96)]
    df = pd.DataFrame([features], columns=cols)
    
    # Select only required features
    df_selected = df[selected_features]
    
    # Predict
    proba = model.predict_proba(df_selected)[0, 1]
    pred = int(proba > 0.5)
    
    # Risk level
    if proba < 0.1:
        risk = "Low"
    elif proba < 0.3:
        risk = "Medium"
    elif proba < 0.5:
        risk = "High"
    else:
        risk = "Critical"
    
    return {
        "bankruptcy_probability": float(proba),
        "prediction": pred,
        "risk_level": risk
    }