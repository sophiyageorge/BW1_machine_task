"""
predict.py

This module exposes a FastAPI application for serving predictions
from a trained Iris classification model. It includes:
- REST endpoints for prediction
- Prometheus metrics for monitoring
- Structured logging for observability

Author: Your Name
Date: 2026
"""

import os
import time
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

from logging_config.logger import common_logger, individual_logger

# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------

app = FastAPI(title="Iris Prediction API", version="1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.pkl")

# -----------------------------------------------------------------------------
# Load Model (with error handling)
# -----------------------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    common_logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    common_logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")


# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------

REQUEST_COUNT = Counter("request_count", "Total API Requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")


# -----------------------------------------------------------------------------
# Request Schema
# -----------------------------------------------------------------------------

class IrisInput(BaseModel):
    """
    Input schema for Iris prediction.
    """
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def home() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: API status message
    """
    return {"message": "ML Model API Running 🚀"}


@app.get("/metrics")
def metrics() -> Response:
    """
    Expose Prometheus metrics.

    Returns:
        Response: Metrics in Prometheus format
    """
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(data: IrisInput) -> dict:
    """
    Predict the Iris class based on input features.

    Args:
        data (IrisInput): Input features

    Returns:
        dict: Predicted class label
    """
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        # Convert input to numpy array
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Make prediction
        prediction = model.predict(features)
        prediction_list: List[int] = prediction.tolist()

        # Individual tracking: detailed logs
        individual_logger.info(
            f"Prediction request | Input: {features.tolist()} | Output: {prediction_list}"
        )

        # Common tracking: aggregate logs
        common_logger.info("Prediction successful")

        return {"prediction": prediction_list}

    except Exception as e:
        common_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        # Record latency regardless of success/failure
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        common_logger.info(f"Request latency: {latency:.4f} seconds")