import sys
import os
from unittest.mock import patch
import numpy as np
from app import app
from fastapi.testclient import TestClient


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
