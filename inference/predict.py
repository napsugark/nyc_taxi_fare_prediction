from contextlib import asynccontextmanager
import io
import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import joblib
import pandas as pd
import traceback
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import requests

from inference.prepare_features import prepare_features_for_prediction


MODEL_URL = "https://dagshub.com/napsugar.kelemen/nyc_taxi_fare_prediction/raw/main/models/model.pkl"
PREPROCESSOR_URL = "https://dagshub.com/napsugar.kelemen/nyc_taxi_fare_prediction/raw/main/models/preprocessor.pkl"

def load_joblib_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    model = load_joblib_from_url(MODEL_URL)
    preprocessor = load_joblib_from_url(PREPROCESSOR_URL)
    yield
    # Optional: code to run on shutdown

app = FastAPI(title="Model Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    pickup_datetime: datetime = Field(..., description="UTC datetime of pickup")
    pickup_longitude: float = Field(..., ge=-75, le=-70, description="Longitude in NYC range")
    pickup_latitude: float = Field(..., ge=35, le=45, description="Latitude in NYC range")
    dropoff_longitude: float = Field(..., ge=-75, le=-70, description="Longitude in NYC range")
    dropoff_latitude: float = Field(..., ge=35, le=45, description="Latitude in NYC range")
    passenger_count: int = Field(..., ge=1, le=6, description="Number of passengers (1-6)")

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2025-06-11T14:30:00Z",
                "pickup_longitude": -73.985428,
                "pickup_latitude": 40.748817,
                "dropoff_longitude": -73.985135,
                "dropoff_latitude": 40.758896,
                "passenger_count": 2
            }
        }

class PredictionResponse(BaseModel):
    prediction: List[float]

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])
        features = prepare_features_for_prediction(input_df)

        # Apply preprocessing
        X_transformed = preprocessor.transform(features)

        # Predict
        prediction = model.predict(X_transformed)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "service": "Taxi Fare Prediction API",
        "status": "ok",
        "description": "This API predicts the fare of a taxi ride based on input parameters like pickup/dropoff location, time, and passenger count.",
        "endpoints": {
            "/predict": "POST request to get fare prediction. Requires pickup/dropoff coordinates, datetime, passenger count, etc.",
            "/health": "GET request to check API health status."
        }
    }