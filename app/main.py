"""
FastAPI Application for Safety Score Model
Crime prediction API with XGBoost model
"""

import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LocationPredictionRequest,
    LocationPredictionResponse
)
from app.predictor import predictor
from app.geo_utils import grid_mapper
from app.datetime_utils import parse_datetime_string, get_time_bin_description
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Loads model on startup
    """
    # Startup: Load the model and grids
    logger.info("Starting up application...")
    logger.info(f"Python version check passed")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        logger.info("Loading model...")
        predictor.load_model()
        logger.info("Model loaded successfully during startup")
        
        logger.info("Loading grid shapefile...")
        grid_mapper.load_grids()
        logger.info("Grid shapefile loaded successfully during startup")
        
        logger.info("✅ Application startup complete - ready to accept requests")
    except Exception as e:
        logger.error(f"❌ Failed to load resources during startup: {str(e)}")
        logger.exception("Full traceback:")
        # Don't raise - let app start anyway for health check
        logger.warning("Starting app despite startup errors")
    
    yield
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Crime Prediction API
    
    Spatio-temporal crime prediction system using XGBoost.
    
    ### Features:
    - Predict crime likelihood for specific grid cells and time windows
    - Binary classification (Safe/Unsafe) with probability scores
    - Support for single and batch predictions
    - Configurable prediction threshold
    
    ### Model Details:
    - Algorithm: XGBoost Binary Classification
    - Target: Unsafe grid cell prediction (0=Safe, 1=Unsafe)
    - Threshold: 0.3 (optimized for high recall)
    - Time granularity: 4-hour bins
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=dict,
    summary="Root endpoint",
    tags=["General"]
)
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "message": "Safety Score Model API",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["General"]
)
async def health_check():
    """
    Health check endpoint
    Returns API status and model loading status
    """
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        model_loaded=predictor.is_loaded,
        version=settings.app_version
    )


@app.get(
    "/model/info",
    summary="Get model information",
    tags=["Model"]
)
async def get_model_info():
    """
    Get information about the loaded model
    Returns model configuration and metadata
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()


@app.post(
    "/model/threshold",
    summary="Update prediction threshold",
    tags=["Model"]
)
async def update_threshold(threshold: float):
    """
    Update the prediction threshold
    
    Args:
        threshold: New threshold value (0.0 - 1.0)
    
    Default threshold is 0.3 (optimized for recall)
    """
    try:
        predictor.set_threshold(threshold)
        return {
            "message": "Threshold updated successfully",
            "new_threshold": threshold
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post(
    f"{settings.api_prefix}/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_single(request: PredictionRequest):
    """
    Make a single crime prediction
    
    Predicts whether a specific grid cell will be unsafe during a given time window
    
    ### Input Features:
    - **index_right**: Grid cell identifier
    - **time_bin**: 4-hour time window (0-5)
    - **day_of_week**: Day of week (0=Mon, 6=Sun)
    - **month**: Month (1-12)
    - **year**: Year
    - **is_weekend**: Weekend indicator (0/1)
    - **lag_1**: Crime count in previous 4-hour window
    - **lag_6**: Crime count 24 hours ago
    - **lag_42**: Crime count 7 days ago
    - **rolling_6_mean**: 24-hour rolling mean
    - **rolling_42_mean**: 7-day rolling mean
    
    ### Returns:
    - **unsafe_probability**: Probability of being unsafe (0.0-1.0)
    - **prediction**: Binary prediction (0=Safe, 1=Unsafe)
    """
    try:
        result = predictor.predict(request)
        logger.info(
            f"Prediction made for grid {request.index_right}: "
            f"prob={result.unsafe_probability:.4f}, pred={result.prediction}"
        )
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    f"{settings.api_prefix}/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple grid cells
    
    Accepts up to 1000 prediction requests at once
    
    ### Returns:
    - **predictions**: List of prediction results
    - **count**: Number of predictions made
    """
    try:
        results = predictor.predict_batch(request.predictions)
        logger.info(f"Batch prediction completed: {len(results)} samples")
        
        return BatchPredictionResponse(
            predictions=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post(
    f"{settings.api_prefix}/predict/location",
    response_model=LocationPredictionResponse,
    summary="Location-based prediction (Simplified)",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_from_location(request: LocationPredictionRequest):
    """
    Make crime prediction from location and datetime (SIMPLIFIED API)
    
    This is the recommended endpoint for frontend applications.
    Simply provide current location coordinates and datetime.
    
    ### Input:
    - **latitude**: Latitude coordinate (e.g., 37.7749)
    - **longitude**: Longitude coordinate (e.g., -122.4194)
    - **datetime**: ISO format datetime string (YYYY-MM-DDTHH:MM:SS)
    
    ### Backend automatically handles:
    - Converting lat/long to grid ID
    - Extracting temporal features (time_bin, day_of_week, etc.)
    - Setting historical features to defaults
    
    ### Returns:
    - **unsafe_probability**: Risk score (0.0-1.0)
    - **prediction**: Binary result (0=Safe, 1=Unsafe)
    - **safety_level**: Human-readable safety level
    - **location**: Location details including grid ID
    - **time_context**: Temporal context information
    """
    try:
        # Check if model is loaded
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Check if grids are loaded
        if not grid_mapper.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grid mapper not loaded. Location service unavailable. Please try again later."
            )
        
        # Parse datetime string
        try:
            dt = parse_datetime_string(request.datetime)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid datetime format: {str(e)}"
            )
        
        # Validate coordinates are within grid bounds
        if not grid_mapper.validate_coordinates(request.latitude, request.longitude):
            grid_info = grid_mapper.get_grid_info()
            bounds = grid_info.get('bounds', {})
            
            # Build bounds message with proper null handling
            if bounds:
                min_lat = bounds.get('min_latitude', 'N/A')
                max_lat = bounds.get('max_latitude', 'N/A')
                min_lon = bounds.get('min_longitude', 'N/A')
                max_lon = bounds.get('max_longitude', 'N/A')
                
                # Format numbers if they exist
                if isinstance(min_lat, (int, float)):
                    min_lat = f"{min_lat:.4f}"
                if isinstance(max_lat, (int, float)):
                    max_lat = f"{max_lat:.4f}"
                if isinstance(min_lon, (int, float)):
                    min_lon = f"{min_lon:.4f}"
                if isinstance(max_lon, (int, float)):
                    max_lon = f"{max_lon:.4f}"
                    
                bounds_text = f"Lat [{min_lat}, {max_lat}], Lon [{min_lon}, {max_lon}]"
            else:
                bounds_text = "Service area bounds unavailable"
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Coordinates ({request.latitude}, {request.longitude}) "
                    f"are outside the service area. "
                    f"Service bounds: {bounds_text}"
                )
            )
        
        # Make prediction
        result = predictor.predict_from_location(
            latitude=request.latitude,
            longitude=request.longitude,
            dt=dt,
            grid_mapper=grid_mapper
        )
        
        # Get grid ID for response
        grid_id = grid_mapper.get_grid_id(request.latitude, request.longitude)
        
        # Extract temporal features for response
        from app.datetime_utils import extract_temporal_features
        temporal_features = extract_temporal_features(dt)
        
        # Determine safety level
        prob = result.unsafe_probability
        if prob < 0.3:
            safety_level = "Low Risk"
        elif prob < 0.5:
            safety_level = "Moderate Risk"
        elif prob < 0.7:
            safety_level = "High Risk"
        else:
            safety_level = "Very High Risk"
        
        # Get day name
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = day_names[temporal_features['day_of_week']]
        
        # Build enhanced response
        response = LocationPredictionResponse(
            unsafe_probability=result.unsafe_probability,
            prediction=result.prediction,
            safety_level=safety_level,
            location={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "grid_id": grid_id
            },
            time_context={
                "datetime": request.datetime,
                "time_period": get_time_bin_description(temporal_features['time_bin']),
                "day": day_name,
                "is_weekend": bool(temporal_features['is_weekend'])
            },
            details={
                "time_bin": temporal_features['time_bin'],
                "day_of_week": temporal_features['day_of_week'],
                "month": temporal_features['month'],
                "year": temporal_features['year']
            }
        )
        
        logger.info(
            f"Location prediction: ({request.latitude}, {request.longitude}) -> "
            f"Grid {grid_id}, prob={result.unsafe_probability:.4f}, {safety_level}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
