from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import joblib
import pandas as pd
import traceback
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

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
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

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