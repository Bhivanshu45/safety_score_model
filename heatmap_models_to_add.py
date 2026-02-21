# ============================================
# Heatmap Models (Crime Safety Visualization)
# ============================================

class GridGeometry(BaseModel):
    """GeoJSON Polygon geometry for a grid cell"""
    type: str = Field(default="Polygon", description="GeoJSON type")
    coordinates: list = Field(
        ...,
        description="Array of coordinate rings [[lon, lat], ...]"
    )


class HeatmapGridData(BaseModel):
    """Single grid cell data for heatmap"""
    grid_id: int = Field(..., description="Unique grid identifier")
    safety_score: float = Field(
        ...,
        description="Safety score (0.0-1.0), higher = more unsafe",
        ge=0.0,
        le=1.0
    )
    safety_level: str = Field(
        ...,
        description="Safety level: Low Risk / Medium Risk / High Risk"
    )
    color: str = Field(
        ...,
        description="Hex color code for visualization"
    )
    geometry: GridGeometry = Field(
        ...,
        description="GeoJSON polygon coordinates"
    )


class HeatmapStatistics(BaseModel):
    """Overall heatmap statistics"""
    total_grids: int = Field(..., description="Total grid cells")
    safe_grids: int = Field(..., description="Low Risk grids (score < 0.33)")
    medium_risk_grids: int = Field(..., description="Medium Risk grids (0.33-0.67)")
    high_risk_grids: int = Field(..., description="High Risk grids (score >= 0.67)")
    average_safety_score: float = Field(
        ...,
        description="Average safety score across all grids",
        ge=0.0,
        le=1.0
    )


class HeatmapResponse(BaseModel):
    """Complete heatmap response with all grid data"""
    timestamp: str = Field(
        ...,
        description="ISO format timestamp when heatmap was generated"
    )
    statistics: HeatmapStatistics = Field(
        ...,
        description="Aggregate statistics for heatmap"
    )
    grids: list[HeatmapGridData] = Field(
        ...,
        description="Array of all grid cells with safety data and geometry"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-02-20T14:30:00",
                "statistics": {
                    "total_grids": 1041,
                    "safe_grids": 450,
                    "medium_risk_grids": 350,
                    "high_risk_grids": 241,
                    "average_safety_score": 0.45
                },
                "grids": [
                    {
                        "grid_id": 1,
                        "safety_score": 0.45,
                        "safety_level": "Low Risk",
                        "color": "#00ff00",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-122.4194, 37.7749], [-122.4185, 37.7749], [-122.4185, 37.7758], [-122.4194, 37.7758], [-122.4194, 37.7749]]]
                        }
                    }
                ]
            }
        }
