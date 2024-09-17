# src/main/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath('src'))

from data_utils import preprocess_data, load_datasets

app = FastAPI()

# Load the pre-trained model
model = joblib.load(r'F:\sem9\assessment\src\model.pkl')  # Raw string

class PredictionRequest(BaseModel):
    features: dict

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert the features into a DataFrame
    features_df = pd.DataFrame([request.features])
    
    # Preprocess the features
    X_processed, _ = preprocess_data(features_df)
    
    # Make prediction
    prediction = model.predict(X_processed)
    return {"prediction": prediction[0]}
