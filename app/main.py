from fastapi import FastAPI, HTTPException
from .schemas import (FinancialRatios, PredictionResponse, 
                      BatchPredictionRequest, BatchPredictionResponse)
from .model_handler import predict_single, load_model

app = FastAPI(
    title="Bankruptcy Prediction API",
    description="Predict company bankruptcy risk from 95 financial ratios.",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Model not loaded on startup: {e}")


@app.get("/")
async def root():
    return {
        "service": "Bankruptcy Prediction API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        load_model()
        return {"status": "healthy", "model_loaded": True}
    except:
        return {"status": "unhealthy", "model_loaded": False}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: FinancialRatios):
    """
    Predict bankruptcy probability for a single company.
    
    Provide exactly 95 financial ratios.
    """
    if len(data.features) != 95:
        raise HTTPException(
            status_code=400, 
            detail=f"Exactly 95 features required, got {len(data.features)}"
        )
    
    try:
        result = predict_single(data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: BatchPredictionRequest):
    """Predict for multiple companies at once."""
    predictions = []
    for sample in data.samples:
        result = predict_single(sample.features)
        predictions.append(result)
    return {"predictions": predictions}