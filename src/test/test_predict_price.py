import sys
import os
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

# Fix import path to include src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Adjust if your app file/module is different
from fastapi.testclient import TestClient

client = TestClient(app)

# Example HousingInput schema for test input (adjust according to your actual schema)
class HousingInput:
    def __init__(self, features):
        self.features = features

    def model_dump_json(self):
        # Return a JSON-serializable dict (simulate Pydantic model method)
        return {"features": self.features}


def test_predict_price_success():
    # Prepare mock input matching your HousingInput schema
    input_data = {"features": [[3, 2, 1200]]}

    # Prepare a fake numpy prediction output
    fake_prediction = np.array([250000])

    # Patch model.predict and log_request inside your app module
    with patch("src.app.model.predict", return_value=fake_prediction) as mock_predict, \
         patch("src.app.log_request") as mock_log:

        response = client.post("/predict", json=input_data)

        assert response.status_code == 200
        assert response.json() == {"predictions": [250000]}
        mock_predict.assert_called_once()
        mock_log.assert_called_once()


def test_predict_price_failure():
    input_data = {"features": [[3, 2, 1200]]}

    # Patch model.predict to raise an Exception
    with patch("src.app.model.predict", side_effect=Exception("Prediction error")):
        response = client.post("/predict", json=input_data)

        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction error"}
