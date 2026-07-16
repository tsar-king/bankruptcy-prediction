"""
Feature selection using Boruta and SHAP importance.
"""
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, 
                              perc: int = 100, random_state: int = 42) -> list:
    """
    Select features using Boruta algorithm.
    
    Boruta is an all-relevant feature selection method that finds all features
    carrying useful information, not just the top-k.
    
    Args:
        X_train: Training features (DataFrame).
        y_train: Training labels.
        perc: Percentile for shadow features (100 = default).
        random_state: Random seed.
    
    Returns:
        List of selected feature names.
    """
    print("Running Boruta feature selection...")
    print(f"  Initial features: {X_train.shape[1]}")
    
    # Use RandomForest as the base estimator
    # class_weight='balanced' handles imbalance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        class_weight='balanced',
        random_state=random_state
    )
    
    # Boruta with automatic number of estimators
    boruta = BorutaPy(
        rf,
        n_estimators='auto',
        perc=perc,
        verbose=1,
        random_state=random_state
    )
    
    # Boruta expects numpy arrays
    boruta.fit(X_train.values, y_train.values)
    
    # Get selected features
    selected_features = X_train.columns[boruta.support_].tolist()
    
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Rejected features: {X_train.shape[1] - len(selected_features)}")
    
    return selected_features


def shap_feature_selection(X_train, y_train, top_k=30, random_state=42):
    """
    Alternative: Select top-k features by SHAP importance from XGBoost.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        top_k: Number of features to keep.
        random_state: Random seed.
    
    Returns:
        List of selected feature names.
    """
    import shap
    from xgboost import XGBClassifier
    
    print("Running SHAP-based feature selection...")
    
    # Train a quick XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Mean absolute SHAP per feature
    importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance['feature'].head(top_k).tolist()
    
    print(f"  Selected top {top_k} features by SHAP importance")
    return selected_features


if __name__ == "__main__":
    from data_loader import load_data, split_data
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Test Boruta
    selected = boruta_feature_selection(X_train, y_train)
    print(f"\nSelected features: {selected}")