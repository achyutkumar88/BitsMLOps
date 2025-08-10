# app.py
from fastapi import FastAPI, HTTPException
from schema.housing_input import HousingInput
import mlflow.pyfunc
import numpy as np
from logger import log_request
from monitoring.prometheus_metrics import setup_prometheus
from contextlib import asynccontextmanager

# Local path to MLflow model (adjust if needed)
core_path = "models/mlruns/565482707616561056/models/"
MODEL_PATH = core_path+"m-352f75c998244a0a8c1f594cf5d6d7cb/artifacts"


# Load model from local path
model = mlflow.pyfunc.load_model(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_prometheus(app)
    yield


# Define FastAPI app
app = FastAPI(title="California Housing Price Predictor")

setup_prometheus(app)


@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is running."}


@app.post("/predict")
def predict_price(input_data: HousingInput):
    try:
        print("Request received -------------->")
        input_array = np.array(input_data.features)
        predictions = model.predict(input_array)
        log_request(input_data.model_dump_json(), predictions.tolist())
        print("Request Processed -------------->")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
