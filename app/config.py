"""
Configuration settings for the Safety Score Model API
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    app_name: str = "Safety Score Model API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Model Settings
    model_file_path: str = os.path.join(
        Path(__file__).parent.parent, 
        "models", 
        "crime_grid_xgb_model.pkl"
    )
    prediction_threshold: float = 0.3
    
    # Feature Configuration
    feature_columns: list = [
        'index_right',
        'time_bin',
        'day_of_week',
        'month',
        'year',
        'is_weekend',
        'lag_1',
        'lag_6',
        'lag_42',
        'rolling_6_mean',
        'rolling_42_mean'
    ]
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("ENVIRONMENT", "development") == "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()
