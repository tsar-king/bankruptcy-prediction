"""
Model training with nested cross-validation, Optuna hyperparameter tuning,
and MLflow experiment tracking.
"""
import mlflow
import mlflow.sklearn
import optuna
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Nested Cross-Validation with Optuna
# ---------------------------------------------------------------------------
def run_nested_cv(X_train, y_train, selected_features, 
                  n_outer_splits=5, n_inner_splits=3,
                  n_trials=30, random_state=42,
                  experiment_name="bankruptcy_nested_cv"):
    """
    Perform nested cross-validation: outer for evaluation, inner for tuning.
    
    Args:
        X_train: Training features (DataFrame).
        y_train: Training labels (Series).
        selected_features: List of feature names to use.
        n_outer_splits: Number of outer CV folds.
        n_inner_splits: Number of inner CV folds.
        n_trials: Number of Optuna trials per fold.
        random_state: Random seed.
        experiment_name: MLflow experiment name.
    
    Returns:
        best_model: Best pipeline from outer folds.
        mean_auc: Mean AUC across outer folds.
        std_auc: Standard deviation of AUC.
    """
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Keep only selected features
    X = X_train[selected_features]
    
    print(f"\n{'='*60}")
    print(f"Starting Nested Cross-Validation")
    print(f"{'='*60}")
    print(f"Features: {len(selected_features)}")
    print(f"Samples: {len(X)}")
    print(f"Bankruptcy rate: {y_train.mean()*100:.2f}%")
    print(f"Outer CV: {n_outer_splits} folds, Inner CV: {n_inner_splits} folds")
    print(f"Optuna trials per fold: {n_trials}")
    print(f"{'='*60}\n")
    
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, 
                               random_state=random_state)
    auc_scores = []
    best_models = []
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="nested_cv_parent") as parent_run:
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y_train)):
            print(f"\n--- Outer Fold {fold_idx+1}/{n_outer_splits} ---")
            
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")
            print(f"  Val bankrupt: {y_val.mean()*100:.2f}%")
            
            # Inner optimization function for Optuna
            def objective(trial):
                # Model choice (you can add LightGBM, RF options)
                model_type = trial.suggest_categorical('model', ['xgboost'])
                
                if model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10, 50),
                        'use_label_encoder': False,
                        'eval_metric': 'logloss',
                        'random_state': random_state
                    }
                    model = XGBClassifier(**params)
                else:
                    # Placeholder for other models
                    pass
                
                # Inner pipeline with SMOTE
                pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=random_state, k_neighbors=3)),
                    ('classifier', model)
                ])
                
                # Inner CV for hyperparameter evaluation
                inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, 
                                           random_state=random_state)
                scores = cross_val_score(pipeline, X_tr, y_tr, 
                                         scoring='roc_auc', cv=inner_cv, 
                                         n_jobs=-1)
                return np.mean(scores)
            
            # Run Optuna
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=random_state),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )
            
            # Suppress Optuna logs for cleaner output
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
            
            # Get best parameters
            best_params = study.best_params
            model_type = best_params.pop('model', 'xgboost')
            
            # Rebuild best model without 'model' param
            if model_type == 'xgboost':
                best_classifier = XGBClassifier(**best_params)
            
            best_pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=random_state, k_neighbors=3)),
                ('classifier', best_classifier)
            ])
            
            # Train on full inner training data
            best_pipeline.fit(X_tr, y_tr)
            
            # Evaluate on outer validation fold
            y_pred_proba = best_pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)
            best_models.append(best_pipeline)
            
            print(f"  Best inner AUC: {study.best_value:.4f}")
            print(f"  Outer validation AUC: {auc:.4f}")
            
            # Log to MLflow (nested run)
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                mlflow.log_params(best_params)
                mlflow.log_metric("outer_val_auc", auc)
                mlflow.log_metric("best_inner_auc", study.best_value)
                try:
                    mlflow.sklearn.log_model(best_pipeline, f"model_fold_{fold_idx}")
                except Exception as e:
                    print(f"  [Warning] Could not log model for fold {fold_idx}: {e}")
    
    # Summary
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print(f"\n{'='*60}")
    print(f"Nested CV Results:")
    print(f"  AUC scores: {[f'{a:.4f}' for a in auc_scores]}")
    print(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"{'='*60}\n")
    
    # Log summary to parent run
    with mlflow.start_run(run_id=parent_run.info.run_id):
        mlflow.log_metrics({
            "cv_mean_auc": mean_auc,
            "cv_std_auc": std_auc,
            "cv_n_features": len(selected_features)
        })
    
    # Return best model (from fold with highest AUC)
    best_fold_idx = np.argmax(auc_scores)
    best_model = best_models[best_fold_idx]
    
    return best_model, mean_auc, std_auc


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------
def train_final_model(X_train, y_train, selected_features, 
                      n_trials=50, random_state=42):
    """
    Train final model on full training data with Optuna tuning.
    
    Args:
        X_train: Training features (DataFrame).
        y_train: Training labels (Series).
        selected_features: Feature names to use.
        n_trials: Optuna trials.
        random_state: Random seed.
    
    Returns:
        final_model: Trained pipeline.
    """
    X = X_train[selected_features]
    
    print(f"\n{'='*60}")
    print(f"Training Final Model")
    print(f"{'='*60}")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10, 50),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': random_state
        }
        
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state, k_neighbors=3)),
            ('classifier', XGBClassifier(**params))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
        return np.mean(scores)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    best_params = study.best_params
    print(f"Best CV AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    
    # Train final model
    final_model = ImbPipeline([
        ('smote', SMOTE(random_state=random_state, k_neighbors=3)),
        ('classifier', XGBClassifier(**best_params))
    ])
    final_model.fit(X, y_train)
    
    # Log and register in MLflow
    with mlflow.start_run(run_name="final_model_production"):
        mlflow.log_params(best_params)
        mlflow.log_metric("final_cv_auc", study.best_value)
        
        # Log the model
        mlflow.sklearn.log_model(final_model, "model")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, "bankruptcy_predictor")
        
        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="bankruptcy_predictor",
            version=registered_model.version,
            stage="Production"
        )
        print(f"Model registered as 'bankruptcy_predictor' v{registered_model.version} (Production)")
    
    # Also save model locally for the API
    joblib.dump(final_model, 'models/final_model.pkl')
    
    return final_model