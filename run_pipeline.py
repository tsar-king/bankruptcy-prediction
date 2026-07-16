"""
Complete pipeline execution script.
Run with: python run_pipeline.py
"""
import os
os.environ['MLFLOW_SKLEARN_USE_SKOPS'] = 'false'
import sys
import warnings
warnings.filterwarnings('ignore')

# Start MLflow tracking server before running this script:
# In another terminal: mlflow ui --port 5000

from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data
from src.feature_selection import boruta_feature_selection
from src.train import run_nested_cv, train_final_model
from src.evaluate import (evaluate_on_test_set, generate_shap_analysis, 
                           bootstrap_prediction_intervals)
from src.utils import save_artifacts, save_metrics


def main():
    print("\n" + "="*70)
    print("BANKRUPTCY PREDICTION PIPELINE")
    print("="*70)
    
    # ----------------------------------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------------------------------
    print("\n[STEP 1] Loading data...")
    df = load_data('data/raw/data.csv')
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    # ----------------------------------------------------------------------
    # Step 2: Preprocessing
    # ----------------------------------------------------------------------
    print("\n[STEP 2] Preprocessing...")
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test, use_robust_scaler=False
    )
    
    # ----------------------------------------------------------------------
    # Step 3: Feature Selection
    # ----------------------------------------------------------------------
    print("\n[STEP 3] Feature Selection...")
    selected_features = boruta_feature_selection(
        X_train, y_train, perc=100, random_state=42
    )
    
    if len(selected_features) == 0:
        print("WARNING: Boruta selected 0 features. Using all features.")
        selected_features = [f'X{i}' for i in range(1, 96)]
    
    print(f"Proceeding with {len(selected_features)} features")
    
    # ----------------------------------------------------------------------
    # Step 4: Nested Cross-Validation
    # ----------------------------------------------------------------------
    print("\n[STEP 4] Nested Cross-Validation...")
    print("Make sure MLflow tracking server is running:")
    print("  In another terminal: mlflow ui --port 5000")
    print("  Then visit http://localhost:5000\n")
    
    input("Press Enter to continue...")
    
    best_model, cv_mean_auc, cv_std_auc = run_nested_cv(
        X_train, y_train,
        selected_features=selected_features,
        n_outer_splits=5,
        n_inner_splits=3,
        n_trials=30,
        random_state=42
    )
    
    # ----------------------------------------------------------------------
    # Step 5: Train Final Model
    # ----------------------------------------------------------------------
    print("\n[STEP 5] Training Final Model...")
    final_model = train_final_model(
        X_train, y_train,
        selected_features=selected_features,
        n_trials=50,
        random_state=42
    )
    
    # ----------------------------------------------------------------------
    # Step 6: Evaluate on Test Set
    # ----------------------------------------------------------------------
    print("\n[STEP 6] Test Set Evaluation...")
    metrics = evaluate_on_test_set(final_model, X_test, y_test, selected_features)
    
    # ----------------------------------------------------------------------
    # Step 7: SHAP Analysis
    # ----------------------------------------------------------------------
    print("\n[STEP 7] SHAP Explainability Analysis...")
    explainer, shap_values = generate_shap_analysis(
        final_model, X_train, X_test, selected_features,
        n_background=100, save_dir='reports'
    )
    
    # ----------------------------------------------------------------------
    # Step 8: Bootstrap Prediction Intervals
    # ----------------------------------------------------------------------
    print("\n[STEP 8] Bootstrap Prediction Intervals...")
    lower, upper = bootstrap_prediction_intervals(
        final_model, X_test, selected_features,
        n_iterations=100, random_state=42
    )
    
    # ----------------------------------------------------------------------
    # Step 9: Save Artifacts
    # ----------------------------------------------------------------------
    print("\n[STEP 9] Saving Artifacts...")
    save_artifacts(preprocessor, selected_features, save_dir='models')
    
    # Save final metrics
    all_metrics = {
        'cv_mean_auc': cv_mean_auc,
        'cv_std_auc': cv_std_auc,
        **metrics
    }
    save_metrics(all_metrics, 'reports/metrics.json')
    
    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults Summary:")
    print(f"  Features selected: {len(selected_features)}")
    print(f"  CV AUC (nested): {cv_mean_auc:.4f} (+/- {cv_std_auc:.4f})")
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test Avg Precision: {metrics['test_avg_precision']:.4f}")
    print(f"\nArtifacts saved to 'models/' and 'reports/'")
    print(f"MLflow UI: http://localhost:5000")
    print(f"\nTo run the API:")
    print(f"  uvicorn app.main:app --reload")
    print(f"  Then visit http://localhost:8000/docs")
    print("="*70)


if __name__ == "__main__":
    main()