from pydantic import BaseModel, Field
from typing import List, Optional


class FinancialRatios(BaseModel):
    """Input schema: 95 financial ratios."""
    features: List[float] = Field(
        ..., 
        min_items=95, 
        max_items=95,
        description="List of 95 financial ratios in order X1 to X95"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5] * 95
            }
        }


class PredictionResponse(BaseModel):
    """Output schema."""
    bankruptcy_probability: float
    prediction: int
    risk_level: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "bankruptcy_probability": 0.12,
                "prediction": 0,
                "risk_level": "Low"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    samples: List[FinancialRatios]


class BatchPredictionResponse(BaseModel):
    """Batch prediction output."""
    predictions: List[PredictionResponse]