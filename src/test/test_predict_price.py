import sys
import os
from unittest.mock import patch
import numpy as np
from src.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_predict_price_success():
    input_data = {"features": [[3, 2, 1200]]}
    fake_prediction = np.array([250000])

    with patch("src.app.model.predict", return_value=fake_prediction) as mock_predict, \
         patch("src.app.log_request") as mock_log:

        response = client.post("/predict", json=input_data)

        expected = {"predictions": [250000]}
        assert response.status_code == 200
        assert response.json() == expected
        mock_predict.assert_called_once()
        mock_log.assert_called_once()


def test_predict_price_failure():
    input_data = {"features": [[3, 2, 1200]]}

    with patch("src.app.model.predict", side_effect=Exception("Prediction error")):
        response = client.post("/predict", json=input_data)

        expected = {"detail": "Prediction error"}
        assert response.status_code == 500
        assert response.json() == expected
