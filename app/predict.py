# app/predict.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI
from fastapi.responses import Response
import joblib
import numpy as np
from pydantic import BaseModel
from logging_config.logger import common_logger, individual_logger
import os
import time


app = FastAPI()

model = joblib.load("app/model.pkl")

REQUEST_COUNT = Counter("request_count", "Total API Requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "ML Model API Running"}



@app.post("/predict")
def predict(data: IrisInput):
    REQUEST_COUNT.inc()

    start_time = time.time()

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)

        # Individual tracking: log every prediction
    individual_logger.info(
        f"Input: {features}, Prediction: {prediction.tolist()}"
    )

    # Common tracking: update overall prediction count
    common_logger.info("One prediction made")

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"prediction": prediction.tolist()}