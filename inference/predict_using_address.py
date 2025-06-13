from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import joblib
import pandas as pd
import traceback
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import requests

from inference.prepare_features import prepare_features_for_prediction

# Load model and preprocessor once at startup
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model.pkl")

app = FastAPI(title="Model Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    pickup_datetime: datetime = Field(..., description="UTC datetime of pickup")
    pickup_address: str
    dropoff_address: str
    passenger_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2025-06-11T14:30:00Z",
                "pickup_address": "350 5th Ave, New York, NY 10118",
                "dropoff_address": "1 World Trade Center, New York, NY 10007",
                "passenger_count": 2
            }
        }

class PredictionResponse(BaseModel):
    prediction: List[float]


def geocode_address(address: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "TaxiFareApp/1.0 (contact@example.com)"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"Could not geocode address: {address}")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

@app.post("/predictbyaddress", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        try:
            pickup_lat, pickup_lon = geocode_address(input_data.pickup_address)
            print(f"Pickup coordinates: {pickup_lat}, {pickup_lon}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Pickup address error: {str(e)}")

        try:
            dropoff_lat, dropoff_lon = geocode_address(input_data.dropoff_address)
            print(f"Dropoff coordinates: {dropoff_lat}, {dropoff_lon}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Dropoff address error: {str(e)}")


        # Build dataframe for prediction
        input_df = pd.DataFrame([{
            "pickup_datetime": input_data.pickup_datetime,
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "passenger_count": input_data.passenger_count
        }])
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