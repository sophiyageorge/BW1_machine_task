# app/predict.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("app/model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API Running"}

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}