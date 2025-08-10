from unittest.mock import patch, MagicMock
import sys
import os

# Insert project root in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with patch("mlflow.pyfunc.load_model") as mock_load_model:
    mock_model = MagicMock()
    mock_model.predict.return_value = [123.45]
    mock_load_model.return_value = mock_model

    import app  # import app while patch is active

from fastapi.testclient import TestClient

client = TestClient(app.app)

def test_predict():
    response = client.post("/predict", json={"features": [[1,2,3,4,5,6,7,8]]})
    assert response.status_code == 200
    assert "predictions" in response.json()
