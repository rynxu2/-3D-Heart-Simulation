import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

class TestFastAPI:
    def test_health_check(self):
        # Act
        response = client.get("/api/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cuda_available" in data

    def test_get_heart_params_valid_label(self):
        # Act
        response = client.get("/api/heart-params/normal")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["level"] == "anatomy"
        assert "heartbeat" in data
        assert data["heartbeat"]["pattern"] == "regular"

    def test_get_heart_params_invalid_label(self):
        # Act
        response = client.get("/api/heart-params/invalid_label_xyz")
        
        # Assert
        assert response.status_code == 400
        assert "Invalid label" in response.json()["detail"]
