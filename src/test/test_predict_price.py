import sys
import os
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient

# Add src directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app import app  # noqa: E402

client = TestClient(app)
predict_path = "src.app.model.predict"

def test_predict_price_success():
    """Test successful prediction response."""
    input_data = {"features": [[3, 2, 1200]]}
    fake_prediction = np.array([250000])
    
    with patch(predict_path, return_value=fake_prediction) as mock_predict, \
            patch("src.app.log_request") as mock_log:

        response = client.post("/predict", json=input_data)

        expected = {"predictions": [250000]}
        assert response.status_code == 200
        assert response.json() == expected
        mock_predict.assert_called_once()
        mock_log.assert_called_once()


def test_predict_price_failure():
    """Test prediction endpoint handles prediction exceptions."""
    input_data = {"features": [[3, 2, 1200]]}

    with patch(predict_path, side_effect=Exception("Prediction error")):

        response = client.post("/predict", json=input_data)

        expected = {"detail": "Prediction error"}
        assert response.status_code == 500
        assert response.json() == expected
