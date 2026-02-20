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
