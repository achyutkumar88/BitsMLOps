import sys
import os
from app import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


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
predict_path = "src.app.model.predict"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello"}


with patch("app.mlflow.pyfunc.load_model") as mock_loader:
    mock_model = MagicMock()
    mock_model.predict.return_value = [123.45]  # sample prediction
    mock_loader.return_value = mock_model


def test_predict_success():
    payload = {"features": [[1, 2, 3, 4, 5, 6, 7, 8]]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"predictions": [123.45]}
