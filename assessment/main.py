# src/main/app.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import joblib
from data_utils import preprocess_data, eda_plots

app = FastAPI()

# Load the pre-trained model
model_path = r'F:\sem9\assessment\src\model_mlp.pkl'
model = joblib.load(model_path)

class PredictionRequest(BaseModel):
    features: dict

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Convert the features into a DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Preprocess the features
        X_processed, _ = preprocess_data(features_df)
        
        # Make prediction
        prediction = model.predict(X_processed)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Preprocess the data
        X_processed, y = preprocess_data(df)
        
        # Generate visualizations
        plots = eda_plots(X_processed, y)
        
        return {
            "churn_distribution": StreamingResponse(plots['churn_distribution'], media_type="image/png"),
            "feature_distributions": StreamingResponse(plots['feature_distributions'], media_type="image/png"),
            "correlation_matrix": StreamingResponse(plots['correlation_matrix'], media_type="image/png"),
            "pairplot": StreamingResponse(plots['pairplot'], media_type="image/png"),
            "boxplots": StreamingResponse(plots['boxplots'], media_type="image/png"),
            "scatter_plots": StreamingResponse(plots['scatter_plots'], media_type="image/png")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
