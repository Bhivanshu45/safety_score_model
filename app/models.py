"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Input schema for crime prediction API
    
    Represents a single grid cell at a specific time with historical features
    """
    
    index_right: int = Field(
        ..., 
        description="Grid cell identifier",
        ge=0
    )
    time_bin: int = Field(
        ..., 
        description="4-hour time bin (0-5 representing 0-3h, 4-7h, ..., 20-23h)",
        ge=0,
        le=5
    )
    day_of_week: int = Field(
        ..., 
        description="Day of the week (0=Monday, 6=Sunday)",
        ge=0,
        le=6
    )
    month: int = Field(
        ..., 
        description="Month of the year (1-12)",
        ge=1,
        le=12
    )
    year: int = Field(
        ..., 
        description="Year",
        ge=2000,
        le=2100
    )
    is_weekend: int = Field(
        ..., 
        description="Weekend indicator (0=Weekday, 1=Weekend)",
        ge=0,
        le=1
    )
    lag_1: float = Field(
        ..., 
        description="Crime count in previous 4-hour window",
        ge=0.0
    )
    lag_6: float = Field(
        ..., 
        description="Crime count 24 hours ago (6 bins * 4 hours)",
        ge=0.0
    )
    lag_42: float = Field(
        ..., 
        description="Crime count 7 days ago (42 bins * 4 hours)",
        ge=0.0
    )
    rolling_6_mean: float = Field(
        ..., 
        description="24-hour rolling mean crime count",
        ge=0.0
    )
    rolling_42_mean: float = Field(
        ..., 
        description="7-day rolling mean crime count",
        ge=0.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "index_right": 15,
                "time_bin": 3,
                "day_of_week": 5,
                "month": 2,
                "year": 2003,
                "is_weekend": 1,
                "lag_1": 0.0,
                "lag_6": 0.0,
                "lag_42": 0.0,
                "rolling_6_mean": 0.0,
                "rolling_42_mean": 0.0
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple grid cells"""
    
    predictions: list[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_length=1,
        max_length=1000
    )


class PredictionResponse(BaseModel):
    """
    Output schema for crime prediction API
    """
    
    unsafe_probability: float = Field(
        ..., 
        description="Probability of the grid cell being unsafe (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    prediction: int = Field(
        ..., 
        description="Binary prediction (0=Safe, 1=Unsafe)",
        ge=0,
        le=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "unsafe_probability": 0.67,
                "prediction": 1
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    count: int = Field(
        ...,
        description="Number of predictions made"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


# ============================================
# Simplified Location-Based Prediction Models
# ============================================

class LocationPredictionRequest(BaseModel):
    """
    Simplified input schema for location-based crime prediction
    
    Frontend only needs to send current location and time
    """
    
    latitude: float = Field(
        ...,
        description="Latitude coordinate",
        ge=-90.0,
        le=90.0,
        json_schema_extra={"example": 37.7749}
    )
    longitude: float = Field(
        ...,
        description="Longitude coordinate",
        ge=-180.0,
        le=180.0,
        json_schema_extra={"example": -122.4194}
    )
    datetime: str = Field(
        ...,
        description="Date and time in ISO format (YYYY-MM-DDTHH:MM:SS)",
        json_schema_extra={"example": "2026-02-20T14:30:00"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "datetime": "2026-02-20T14:30:00"
            }
        }


class LocationPredictionResponse(BaseModel):
    """
    Enhanced output schema with location context
    """
    
    # Prediction results
    unsafe_probability: float = Field(
        ..., 
        description="Probability of the location being unsafe (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    prediction: int = Field(
        ..., 
        description="Binary prediction (0=Safe, 1=Unsafe)",
        ge=0,
        le=1
    )
    safety_level: str = Field(
        ...,
        description="Human-readable safety level"
    )
    
    # Location context
    location: dict = Field(
        ...,
        description="Input location coordinates"
    )
    
    # Time context
    time_context: dict = Field(
        ...,
        description="Temporal information"
    )
    
    # Technical details (optional)
    details: Optional[dict] = Field(
        None,
        description="Additional technical details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "unsafe_probability": 0.67,
                "prediction": 1,
                "safety_level": "High Risk",
                "location": {
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "grid_id": 15
                },
                "time_context": {
                    "datetime": "2026-02-20T14:30:00",
                    "time_period": "12:00-15:59 (Afternoon)",
                    "day": "Thursday",
                    "is_weekend": False
                },
                "details": {
                    "time_bin": 3,
                    "day_of_week": 3,
                    "month": 2,
                    "year": 2026
                }
            }
        }

# ============================================
# Route Safety Score Prediction Models
# ============================================

class RouteCoordinate(BaseModel):
    """Single coordinate in a route"""
    lat: float = Field(
        ...,
        description="Latitude coordinate",
        ge=-90.0,
        le=90.0,
        alias="latitude"
    )
    lng: float = Field(
        ...,
        description="Longitude coordinate",
        ge=-180.0,
        le=180.0,
        alias="longitude"
    )
    
    class Config:
        populate_by_name = True  # Allow both lat/latitude and lng/longitude


class RoutePredictionRequest(BaseModel):
    """
    Route safety prediction request
    
    Takes array of coordinates representing path from source to destination.
    Preprocesses each location, runs model predictions, and returns aggregated route safety.
    """
    
    coordinates: list[RouteCoordinate] = Field(
        ...,
        description="Array of coordinates representing the route path",
        min_length=2,
        max_length=500
    )
    datetime: str = Field(
        ...,
        description="ISO format datetime (YYYY-MM-DDTHH:MM:SS)",
        json_schema_extra={"example": "2026-02-20T14:30:00"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "coordinates": [
                    {"lat": 37.7749, "lng": -122.4194},
                    {"lat": 37.7750, "lng": -122.4195},
                    {"lat": 37.7751, "lng": -122.4196}
                ],
                "datetime": "2026-02-20T14:30:00"
            }
        }


class RoutePredictionResponse(BaseModel):
    """
    Route safety prediction response
    
    Returns aggregated safety score for entire route path using both
    average and worst-case aggregation methods.
    """
    
    # Overall route metrics (using average of all segment probabilities)
    route_safety_score: float = Field(
        ...,
        description="Average safety score across all route segments (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    route_safety_level: str = Field(
        ...,
        description="Human-readable safety level (Low Risk/Medium Risk/High Risk)"
    )
    
    # Worst case metric
    worst_probability: float = Field(
        ...,
        description="Maximum (worst-case) unsafe probability in route (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Average metric
    average_probability: float = Field(
        ...,
        description="Average unsafe probability across all segments (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Summary
    segment_count: int = Field(
        ...,
        description="Total number of segments evaluated"
    )
    unsafe_segments: int = Field(
        ...,
        description="Number of segments predicted as unsafe"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "route_safety_score": 0.65,
                "route_safety_level": "Medium Risk",
                "worst_probability": 0.82,
                "average_probability": 0.65,
                "segment_count": 3,
                "unsafe_segments": 1
            }
        }


# ============================================
# Heatmap API Models
# ============================================

class GridGeometry(BaseModel):
    """GeoJSON-compatible polygon geometry for a grid cell"""
    type: str = Field(default="Polygon", description="Geometry type")
    coordinates: list = Field(..., description="Polygon coordinates in [[lon, lat], ...] format")


class HeatmapGridData(BaseModel):
    """Single grid cell data for heatmap visualization"""
    
    grid_id: int = Field(..., description="Unique grid cell identifier")
    unsafe_probability: float = Field(
        ...,
        description="Probability of being unsafe (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    safety_level: str = Field(
        ...,
        description="Safety level: Safe(0), Low Risk(0-0.4), Medium Risk(0.4-0.7), High Risk(0.7-1.0)"
    )
    color: str = Field(
        ...,
        description="Hex color code (#RRGGBB) for map visualization"
    )
    geometry: GridGeometry = Field(..., description="GeoJSON polygon geometry")


class HeatmapStatistics(BaseModel):
    """Overall statistics for heatmap"""
    
    total_grids: int = Field(..., description="Total number of grid cells")
    safe_count: int = Field(..., description="Number of safe grids")
    low_risk_count: int = Field(..., description="Number of low risk grids")
    medium_risk_count: int = Field(..., description="Number of medium risk grids")
    high_risk_count: int = Field(..., description="Number of high risk grids")
    average_unsafe_probability: float = Field(
        ...,
        description="Average unsafe probability across all grids",
        ge=0.0,
        le=1.0
    )


class HeatmapResponse(BaseModel):
    """
    Complete heatmap response for visualization
    
    Returns all grid cells with safety scores, colors, and geometries for map rendering
    """
    
    timestamp: str = Field(
        ...,
        description="ISO format datetime when heatmap was generated"
    )
    statistics: HeatmapStatistics = Field(
        ...,
        description="Overall statistics for the heatmap"
    )
    grids: list[HeatmapGridData] = Field(
        ...,
        description="Array of grid cells with safety data and geometries"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-02-20T14:30:00",
                "statistics": {
                    "total_grids": 1041,
                    "safe_count": 500,
                    "low_risk_count": 300,
                    "medium_risk_count": 150,
                    "high_risk_count": 91,
                    "average_unsafe_probability": 0.35
                },
                "grids": [
                    {
                        "grid_id": 0,
                        "unsafe_probability": 0.25,
                        "safety_level": "Low Risk",
                        "color": "#FFA500",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-122.5, 37.7], [-122.5, 37.75], [-122.45, 37.75], [-122.45, 37.7], [-122.5, 37.7]]]
                        }
                    }
                ]
            }
        }