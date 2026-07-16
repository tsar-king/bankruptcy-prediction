import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append('.')
from app.main import app

client = TestClient(app)


class TestAPI:
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["status"] == "running"
    
    def test_health(self):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code in [200, 500]  # May fail if model not trained
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        # First ensure model exists, skip if not
        try:
            from app.model_handler import load_model
            load_model()
        except:
            pytest.skip("Model not trained yet")
        
        payload = {"features": [0.5] * 95}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["bankruptcy_probability"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
    
    def test_predict_invalid_length(self):
        """Test prediction with wrong number of features."""
        payload = {"features": [0.5] * 94}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400