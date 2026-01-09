"""FastAPI application for dog breed classification."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import yaml
from pathlib import Path

from inference.predictor import load_predictor
from inference.api.schemas import PredictionResponse, HealthResponse, ErrorResponse

# Initialize FastAPI app
app = FastAPI(
    title="Dog Breed Classifier API",
    description="API for classifying dog breeds using genetics-informed deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded on startup)
predictor = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    
    try:
        # Load from inference config
        config_path = Path("config/inference_config.yaml")
        
        if not config_path.exists():
            print("Warning: inference_config.yaml not found, using train_config.yaml")
            config_path = Path("config/train_config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = Path(config['inference']['model_path'])
        
        print(f"Loading model from {model_path}...")
        predictor = load_predictor(
            model_path=model_path,
            config_path=config_path,
            device=config['inference'].get('device', 'auto')
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Dog Breed Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        num_classes=len(predictor.class_names) if predictor else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Predict dog breed from uploaded image.
    
    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return (default: 5)
    
    Returns:
        Prediction results with top-k breeds
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = predictor.predict(image, top_k=top_k)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(
    files: list[UploadFile] = File(...),
    top_k: int = 5
):
    """
    Predict dog breeds for multiple images.
    
    Args:
        files: List of uploaded image files
        top_k: Number of top predictions per image
    
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            images.append(image)
        
        results = predictor.predict_batch(images, top_k=top_k)
        
        return [PredictionResponse(**result) for result in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
