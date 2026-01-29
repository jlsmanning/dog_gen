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


class ExemplarInfo(BaseModel):
    """Information about an available exemplar."""
    class_name: str
    breed_name: str


class ExemplarsResponse(BaseModel):
    """Response model for exemplars list endpoint."""
    count: int
    exemplars: List[ExemplarInfo]


class ErrorAnalysis(BaseModel):
    """Error severity analysis based on genetic distance."""
    genetic_distance: float
    error_severity: str  # "correct", "minor", "moderate", "major"
    description: str


class DemoResponse(BaseModel):
    """Response model for demo endpoint."""
    exemplar_used: ExemplarInfo
    prediction: PredictionResponse
    error_analysis: ErrorAnalysis
