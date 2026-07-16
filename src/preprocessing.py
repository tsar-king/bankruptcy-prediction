"""
Preprocessing pipeline for the bankruptcy dataset.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np


def create_preprocessing_pipeline(use_robust_scaler: bool = False):
    """
    Create a preprocessing pipeline for numeric financial ratios.
    
    Args:
        use_robust_scaler: If True, use RobustScaler (better for outliers).
                          If False, use StandardScaler.
    
    Returns:
        sklearn Pipeline object.
    """
    if use_robust_scaler:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    return pipeline


def preprocess_data(X_train, X_test, use_robust_scaler=False):
    """
    Fit preprocessor on training data and transform both train and test.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        use_robust_scaler: Whether to use RobustScaler.
    
    Returns:
        X_train_processed, X_test_processed, fitted_pipeline
    """
    pipeline = create_preprocessing_pipeline(use_robust_scaler)
    
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"Preprocessing complete. Train shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, pipeline


if __name__ == "__main__":
    from data_loader import load_data, split_data
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_p, X_test_p, pipe = preprocess_data(X_train, X_test)
    
    # Verify no NaNs
    assert not np.isnan(X_train_p).any(), "NaN found in processed training data"
    assert not np.isnan(X_test_p).any(), "NaN found in processed test data"
    print("All checks passed!")