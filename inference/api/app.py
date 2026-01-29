"""FastAPI application for dog breed classification."""

from typing import List
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import yaml
from pathlib import Path

from inference.predictor import load_predictor
from inference.api.schemas import (
    PredictionResponse, HealthResponse, ExemplarInfo,
    ExemplarsResponse, DemoResponse, ErrorAnalysis
)
import numpy as np

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
exemplars_path = None
distance_matrix = None


def compute_error_analysis(true_idx: int, pred_idx: int) -> ErrorAnalysis:
    """Compute error severity based on genetic distance."""
    if true_idx == pred_idx:
        return ErrorAnalysis(
            genetic_distance=0.0,
            error_severity="correct",
            description="Correct prediction"
        )

    dist = float(distance_matrix[true_idx, pred_idx])

    if dist < 0.28:
        severity = "minor"
        description = "Minor error - genetically close breeds (note: genetic similarity may differ from visual similarity)"
    elif dist < 0.35:
        severity = "moderate"
        description = "Moderate error - typical genetic distance between breeds"
    else:
        severity = "major"
        description = "Major error - genetically distant breeds"

    return ErrorAnalysis(
        genetic_distance=dist,
        error_severity=severity,
        description=description
    )


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor, exemplars_path, distance_matrix

    try:
        # Load from inference config
        config_path = Path("config/inference_config.yaml")

        if not config_path.exists():
            print("Warning: inference_config.yaml not found, using train_config.yaml")
            config_path = Path("config/train_config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_path = Path(config['inference']['model_path'])
        class_names_path = config['inference'].get('class_names_path')

        # Load exemplars path
        exemplars_path = Path(config['inference'].get('exemplars_path', 'data/exemplars'))
        if exemplars_path.exists():
            print(f"Exemplars available at {exemplars_path}")
        else:
            print(f"Warning: Exemplars path {exemplars_path} not found")
            exemplars_path = None

        # Load genetic distance matrix
        distances_path = Path(config['inference'].get('distances_path', 'saved_models/genetic_distances.json'))
        if distances_path.exists():
            import json
            with open(distances_path, 'r') as f:
                dist_data = json.load(f)
            distance_matrix = np.array(dist_data['distance_matrix'])
            print(f"Genetic distance matrix loaded ({distance_matrix.shape})")
        else:
            print(f"Warning: Distance matrix {distances_path} not found - error analysis disabled")
            distance_matrix = None

        print(f"Loading model from {model_path}...")
        predictor = load_predictor(
            model_path=model_path,
            config_path=config_path,
            class_names_path=class_names_path,
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
            "demo": "/demo",
            "exemplars": "/exemplars",
            "predict_exemplar": "/predict_exemplar/{breed}",
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


@app.get("/exemplars", response_model=ExemplarsResponse)
async def list_exemplars():
    """
    List all available exemplar breeds for testing.

    Returns:
        List of available breeds with their class and display names
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if exemplars_path is None or not exemplars_path.exists():
        raise HTTPException(status_code=404, detail="Exemplars not available")

    exemplars = []
    for i, class_name in enumerate(predictor.class_names):
        breed_dir = exemplars_path / class_name
        if breed_dir.exists():
            exemplars.append(ExemplarInfo(
                class_name=class_name,
                breed_name=predictor.genetic_names[i]
            ))

    return ExemplarsResponse(count=len(exemplars), exemplars=exemplars)


@app.get("/demo", response_model=DemoResponse)
async def demo():
    """
    Run a demo prediction using a random exemplar image.

    Returns:
        The exemplar used and its prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if exemplars_path is None or not exemplars_path.exists():
        raise HTTPException(status_code=404, detail="Exemplars not available")

    # Get available exemplars
    available = []
    for i, class_name in enumerate(predictor.class_names):
        breed_dir = exemplars_path / class_name
        if breed_dir.exists():
            images = list(breed_dir.glob('*'))
            if images:
                available.append((i, class_name, images))

    if not available:
        raise HTTPException(status_code=404, detail="No exemplar images found")

    # Pick random breed and image
    idx, class_name, images = random.choice(available)
    image_path = random.choice(images)

    try:
        with Image.open(image_path) as img:
            image = img.convert('RGB').copy()

        result = predictor.predict(image, top_k=5)

        # Get predicted class index for error analysis
        pred_class_name = result['top_prediction']['class_name']
        pred_idx = predictor.class_names.index(pred_class_name)

        # Compute error analysis
        error_analysis = compute_error_analysis(idx, pred_idx)

        return DemoResponse(
            exemplar_used=ExemplarInfo(
                class_name=class_name,
                breed_name=predictor.genetic_names[idx]
            ),
            prediction=PredictionResponse(**result),
            error_analysis=error_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo error: {str(e)}")


@app.get("/predict_exemplar/{breed}", response_model=DemoResponse)
async def predict_exemplar(breed: str, top_k: int = 5):
    """
    Predict using an exemplar image for a specific breed.

    Args:
        breed: Either the class_name (e.g., 'n02088364-beagle') or
               breed_name (e.g., 'Beagle') - case insensitive
        top_k: Number of top predictions to return

    Returns:
        The exemplar used and its prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if exemplars_path is None or not exemplars_path.exists():
        raise HTTPException(status_code=404, detail="Exemplars not available")

    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    # Find matching breed (by class_name or genetic_name)
    breed_lower = breed.lower()
    found_idx = None
    found_class = None

    for i, class_name in enumerate(predictor.class_names):
        genetic_name = predictor.genetic_names[i]
        if (class_name.lower() == breed_lower or
            genetic_name.lower() == breed_lower or
            breed_lower in class_name.lower() or
            breed_lower in genetic_name.lower()):
            found_idx = i
            found_class = class_name
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Breed '{breed}' not found")

    # Get exemplar image
    breed_dir = exemplars_path / found_class
    if not breed_dir.exists():
        raise HTTPException(status_code=404, detail=f"No exemplars for breed '{breed}'")

    images = list(breed_dir.glob('*'))
    if not images:
        raise HTTPException(status_code=404, detail=f"No images found for breed '{breed}'")

    image_path = random.choice(images)

    try:
        with Image.open(image_path) as img:
            image = img.convert('RGB').copy()

        result = predictor.predict(image, top_k=top_k)

        # Get predicted class index for error analysis
        pred_class_name = result['top_prediction']['class_name']
        pred_idx = predictor.class_names.index(pred_class_name)

        # Compute error analysis
        error_analysis = compute_error_analysis(found_idx, pred_idx)

        return DemoResponse(
            exemplar_used=ExemplarInfo(
                class_name=found_class,
                breed_name=predictor.genetic_names[found_idx]
            ),
            prediction=PredictionResponse(**result),
            error_analysis=error_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(
    files: List[UploadFile] = File(...),
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
