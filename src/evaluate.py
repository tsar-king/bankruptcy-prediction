"""
Model evaluation: test set metrics, SHAP explainability, bootstrap intervals.
Uses model-agnostic SHAP explainer to avoid XGBoost base_score bug.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from sklearn.base import clone
import joblib
import matplotlib
matplotlib.use('Agg')


def evaluate_on_test_set(model, X_test, y_test, selected_features):
    """
    Evaluate the final model on the hold-out test set.
    """
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation")
    print(f"{'='*60}")

    X = X_test[selected_features]

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Bankrupt', 'Bankrupt']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    metrics = {
        'test_auc': auc,
        'test_avg_precision': avg_precision,
        'confusion_matrix': cm
    }

    # Log to MLflow (gracefully handle permission errors)
    try:
        with mlflow.start_run(run_name="test_evaluation", nested=True):
            mlflow.log_metrics({
                'test_auc': auc,
                'test_avg_precision': avg_precision
            })
    except Exception as e:
        print(f"  [Warning] Could not log to MLflow: {e}")

    return metrics


def _safe_log_artifact(path):
    """Log an artifact to MLflow, ignoring errors (e.g., permission denied)."""
    try:
        mlflow.log_artifact(path)
    except Exception as e:
        print(f"  [Warning] Could not log artifact {path}: {e}")


def generate_shap_analysis(model, X_train, X_test, y_test, selected_features,
                           n_background=100, save_dir='reports',
                           n_shap_samples=200):
    """
    Generate SHAP explanations using a model-agnostic explainer.
    Robust to different explainer output shapes.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating SHAP Analysis (model-agnostic)")
    print(f"{'='*60}")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    X_explain = X_test_sel.iloc[:n_shap_samples]

    background = shap.sample(X_train_sel, n_background, random_state=42)
    masker = shap.maskers.Independent(background)

    print("  Creating model-agnostic SHAP explainer ...")
    explainer = shap.Explainer(
        model.predict_proba,
        masker,
        algorithm="permutation"
    )

    print(f"  Computing SHAP values for {X_explain.shape[0]} samples ...")
    shap_obj = explainer(X_explain)

    # Extract numeric SHAP values for the positive class (bankrupt, index 1)
    vals_all = shap_obj.values
    if isinstance(vals_all, list):                # multi‑output: list of arrays
        pos_vals = np.array(vals_all[1])          # shape (samples, features)
    elif vals_all.ndim == 3:                      # shape (samples, features, classes)
        pos_vals = vals_all[:, :, 1]              # positive class
    else:                                         # shape (samples, features)
        pos_vals = vals_all

    # Compute base value (expected model output over background)
    base_val = np.mean(model.predict_proba(background)[:, 1])

    # --- Summary Plot (Bar) ---
    print("  Generating summary bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(pos_vals, X_explain, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14)
    plt.tight_layout()
    bar_path = f'{save_dir}/shap_summary_bar.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    _safe_log_artifact(bar_path)
    print(f"  Saved: {bar_path}")

    # --- Summary Plot (Beeswarm) ---
    print("  Generating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(pos_vals, X_explain, show=False, max_display=20)
    plt.title("SHAP Beeswarm Plot", fontsize=14)
    plt.tight_layout()
    beeswarm_path = f'{save_dir}/shap_beeswarm.png'
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    _safe_log_artifact(beeswarm_path)
    print(f"  Saved: {beeswarm_path}")

    # --- Waterfall Plot for a Bankruptcy Case ---
    print("  Generating waterfall plot for a bankrupt company...")
    y_explain = y_test.loc[X_explain.index]
    bankrupt_indices = np.where(y_explain.values == 1)[0]
    if len(bankrupt_indices) > 0:
        idx = bankrupt_indices[0]
        # Extract 1D SHAP values for this sample
        waterfall_vals = pos_vals[idx]            # shape (features,)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=waterfall_vals,
                base_values=base_val,
                data=X_explain.iloc[idx].values,
                feature_names=X_explain.columns.tolist()
            ),
            max_display=10,
            show=False
        )
        plt.title(f"SHAP Waterfall - Bankrupt Company (Row {X_explain.index[idx]})", fontsize=14)
        plt.tight_layout()
        waterfall_path = f'{save_dir}/shap_waterfall_bankrupt.png'
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
        plt.close()
        _safe_log_artifact(waterfall_path)
        print(f"  Saved: {waterfall_path}")
    else:
        print("  No bankrupt cases in sampled set, skipping waterfall plot.")

    # --- Dependence Plot for Top Feature ---
    top_feature_idx = np.argmax(np.abs(pos_vals).mean(axis=0))
    top_feature = X_explain.columns[top_feature_idx]
    print(f"  Generating dependence plot for top feature: {top_feature}...")
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, pos_vals, X_explain, show=False)
    plt.title(f"SHAP Dependence: {top_feature}", fontsize=14)
    plt.tight_layout()
    dep_path = f'{save_dir}/shap_dependence_{top_feature}.png'
    plt.savefig(dep_path, dpi=150, bbox_inches='tight')
    plt.close()
    _safe_log_artifact(dep_path)
    print(f"  Saved: {dep_path}")

    print("SHAP analysis complete!")
    return explainer, pos_vals


def bootstrap_prediction_intervals(model, X_test, selected_features,
                                   n_iterations=100, random_state=42):
    """
    Generate prediction intervals using bootstrap resampling.
    """
    print(f"\n{'='*60}")
    print(f"Computing Bootstrap Prediction Intervals")
    print(f"{'='*60}")

    X = X_test[selected_features]
    np.random.seed(random_state)

    all_predictions = []

    for i in range(n_iterations):
        # Bootstrap sample indices
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[indices]

        # Predict
        preds = model.predict_proba(X_boot)[:, 1]
        all_predictions.append(preds)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_iterations} iterations")

    all_predictions = np.array(all_predictions)

    lower = np.percentile(all_predictions, 2.5, axis=0)
    upper = np.percentile(all_predictions, 97.5, axis=0)

    # Visualize for a few samples
    n_show = min(5, len(X))
    plt.figure(figsize=(12, 6))
    for i in range(n_show):
        plt.errorbar(i, np.mean(all_predictions[:, i], axis=0),
                     yerr=[[np.mean(all_predictions[:, i]) - lower[i]],
                           [upper[i] - np.mean(all_predictions[:, i])]],
                     fmt='o', capsize=5, label=f'Sample {i}' if i == 0 else "")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision threshold')
    plt.xlabel('Test Sample')
    plt.ylabel('Predicted Bankruptcy Probability')
    plt.title('Bootstrap Prediction Intervals (95% CI)')
    plt.legend()
    plt.tight_layout()
    ci_path = 'reports/bootstrap_intervals.png'
    plt.savefig(ci_path, dpi=150, bbox_inches='tight')
    plt.close()
    _safe_log_artifact(ci_path)
    print(f"  Saved: {ci_path}")

    print("Bootstrap intervals complete!")
    return lower, upper