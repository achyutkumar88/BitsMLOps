from unittest.mock import patch, MagicMock
import sys
import os
from app import app
from fastapi.testclient import TestClient


# Patch mlflow.pyfunc.load_model before app imports it
patcher = patch("mlflow.pyfunc.load_model")
mock_load_model = patcher.start()
mock_model = MagicMock()
mock_model.predict.return_value = [123.45]
mock_load_model.return_value = mock_model


# Add src directory to sys.path for imports
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)


client = TestClient(app)


def teardown_module(module):
    patcher.stop()


predict_path = "src.app.model.predict"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    msg = "California Housing Prediction API is running."
    assert response.json() == {"message": msg}


def test_predict():
    response = client.post(
    "/predict", json={"features": [[1, 2, 3, 4, 5, 6, 7, 8]]})
    assert response.status_code == 200
    assert "predictions" in response.json()
