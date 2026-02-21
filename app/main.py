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
    LocationPredictionResponse,
    RoutePredictionRequest,
    RoutePredictionResponse,
    HeatmapResponse
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


@app.post(
    f"{settings.api_prefix}/predict/route",
    response_model=RoutePredictionResponse,
    summary="Route safety prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_route(request: RoutePredictionRequest):
    """
    Predict safety score for a complete route/path
    
    Takes an array of coordinates representing a path from source to destination.
    Preprocesses each location, runs model predictions for all points, and returns
    aggregated route safety metrics using both average and worst-case methods.
    
    ### Input:
    - **coordinates**: Array of {lat, lng} coordinate objects (min 2, max 500)
    - **datetime**: ISO format datetime (YYYY-MM-DDTHH:MM:SS)
    
    ### Backend Pipeline:
    1. Convert all lat/lng → grid IDs
    2. Extract temporal features from datetime
    3. Prepare historical lag features
    4. Run batch predictions on all locations
    5. Aggregate using average and worst-case methods
    
    ### Returns:
    - **route_safety_score**: Average safety score (0.0-1.0)
    - **route_safety_level**: Human-readable level (Low/Medium/High Risk)
    - **worst_probability**: Worst-case (max) safety score
    - **average_probability**: Average safety score
    - **segment_count**: Total locations evaluated
    - **unsafe_segments**: Number of locations predicted as unsafe
    """
    try:
        # Validate model and grids are loaded
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        if not grid_mapper.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grid mapper not loaded. Service unavailable."
            )
        
        # Parse datetime
        try:
            dt = parse_datetime_string(request.datetime)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid datetime format: {str(e)}"
            )
        
        # Convert coordinates list to tuples (lat, lng)
        coordinates = [(coord.lat, coord.lng) for coord in request.coordinates]
        
        # Validate coordinates are within bounds
        for idx, (lat, lng) in enumerate(coordinates):
            if not grid_mapper.validate_coordinates(lat, lng):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Coordinate {idx} ({lat}, {lng}) is outside service area bounds"
                )
        
        logger.info(
            f"Route prediction requested: {len(coordinates)} coordinates, "
            f"datetime={request.datetime}"
        )
        
        # Run route prediction
        result = predictor.predict_route(coordinates, dt, grid_mapper)
        
        # Create and return response
        response = RoutePredictionResponse(
            route_safety_score=result['route_safety_score'],
            route_safety_level=result['route_safety_level'],
            worst_probability=result['worst_probability'],
            average_probability=result['average_probability'],
            segment_count=result['segment_count'],
            unsafe_segments=result['unsafe_segments']
        )
        
        logger.info(
            f"Route prediction complete: avg={result['average_probability']:.4f}, "
            f"worst={result['worst_probability']:.4f}, "
            f"unsafe_segments={result['unsafe_segments']}/{result['segment_count']}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route prediction failed: {str(e)}"
        )


@app.get(
    f"{settings.api_prefix}/heatmap",
    response_model=HeatmapResponse,
    summary="Crime safety heatmap",
    tags=["Visualization"],
    status_code=status.HTTP_200_OK
)
async def get_heatmap():
    """
    Get complete crime safety heatmap for all grid cells
    
    Returns safety predictions for all grid cells in the city as a GeoJSON-compatible
    response with color-coded safety levels. This can be rendered as a heatmap overlay
    on a web map.
    
    ### Backend Process:
    1. Predicts safety score for all 1041 grid cells
    2. Uses current datetime for temporal features
    3. Groups grids by risk level (Low/Medium/High)
    4. Assigns colors: Green/Yellow/Red
    5. Returns with polygon geometries for each cell
    
    ### Response includes:
    - **timestamp**: When heatmap was generated
    - **statistics**: Summary of safe/medium/high risk grids
    - **grids**: Array of all grids with safety scores and geometries
    
    ### Frontend Usage:
    ```javascript
    fetch('/api/heatmap')
      .then(res => res.json())
      .then(data => {
        // Render polygons using Leaflet/Mapbox
        data.grids.forEach(grid => {
          L.geoJSON(grid.geometry, {
            style: { color: grid.color }
          }).addTo(map);
        });
      });
    ```
    """
    try:
        # Validate model and grids are loaded
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        if not grid_mapper.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Grid mapper not loaded. Service unavailable."
            )
        
        logger.info("Heatmap request received")
        
        # Run heatmap prediction for all grids
        result = predictor.predict_all_grids(grid_mapper)
        
        # Create response object
        response = HeatmapResponse(
            timestamp=result['timestamp'],
            statistics={
                "total_grids": result['statistics']['total_grids'],
                "safe_count": result['statistics']['safe_count'],
                "low_risk_count": result['statistics']['low_risk_count'],
                "medium_risk_count": result['statistics']['medium_risk_count'],
                "high_risk_count": result['statistics']['high_risk_count'],
                "average_unsafe_probability": result['statistics']['average_unsafe_probability']
            },
            grids=result['grids']
        )
        
        logger.info(
            f"Heatmap complete: {result['statistics']['total_grids']} grids, "
            f"Safe={result['statistics']['safe_count']}, "
            f"Low Risk={result['statistics']['low_risk_count']}, "
            f"Medium={result['statistics']['medium_risk_count']}, "
            f"High={result['statistics']['high_risk_count']}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heatmap generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heatmap generation failed: {str(e)}"
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
