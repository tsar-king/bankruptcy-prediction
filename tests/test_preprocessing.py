import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from src.preprocessing import create_preprocessing_pipeline, preprocess_data
from src.data_loader import load_data, split_data


class TestPreprocessing:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.df = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)
    
    def test_pipeline_creation(self):
        """Test preprocessing pipeline can be created."""
        pipe = create_preprocessing_pipeline()
        assert pipe is not None
        assert len(pipe.steps) == 2  # imputer + scaler
    
    def test_transform_shape(self):
        """Test preprocessing maintains shape."""
        X_train_p, X_test_p, pipe = preprocess_data(self.X_train, self.X_test)
        assert X_train_p.shape[0] == self.X_train.shape[0]
        assert X_train_p.shape[1] == self.X_train.shape[1]
        assert X_test_p.shape[0] == self.X_test.shape[0]
    
    def test_no_nan_after_transform(self):
        """Test no NaN values after preprocessing."""
        X_train_p, X_test_p, pipe = preprocess_data(self.X_train, self.X_test)
        assert not np.isnan(X_train_p).any()
        assert not np.isnan(X_test_p).any()
    
    def test_scaling(self):
        """Test that data is scaled (mean ≈ 0, std ≈ 1)."""
        X_train_p, X_test_p, pipe = preprocess_data(self.X_train, self.X_test)
        means = np.mean(X_train_p, axis=0)
        stds = np.std(X_train_p, axis=0)
        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(stds, 1, atol=1e-5)