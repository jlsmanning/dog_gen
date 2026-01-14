"""Pydantic schemas for API request/response models."""

from typing import List, Optional
from pydantic import BaseModel


class BreedPrediction(BaseModel):
    """Single breed prediction."""
    class_name: str
    breed_name: str
    probability: float
    confidence: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""
    top_prediction: BreedPrediction
    top_k_predictions: List[BreedPrediction]


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool
    num_classes: Optional[int] = None


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    detail: str
