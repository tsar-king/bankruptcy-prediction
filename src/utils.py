"""
Utility functions for saving/loading artifacts and logging.
"""
import joblib
import json
import os
from datetime import datetime


def save_artifacts(preprocessor, selected_features, save_dir='models'):
    """
    Save preprocessor and feature list for API use.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(preprocessor, f'{save_dir}/preprocessor.pkl')
    joblib.dump(selected_features, f'{save_dir}/selected_features.pkl')
    
    print(f"Artifacts saved to {save_dir}/")


def save_metrics(metrics, path='reports/metrics.json'):
    """Save evaluation metrics as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to native Python
    metrics_serializable = {}
    for k, v in metrics.items():
        if hasattr(v, 'tolist'):
            metrics_serializable[k] = v.tolist()
        elif hasattr(v, 'item'):
            metrics_serializable[k] = v.item()
        else:
            metrics_serializable[k] = v
    
    with open(path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Metrics saved to {path}")


def get_timestamp():
    """Return formatted timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")