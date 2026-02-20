"""
Geospatial Utilities
Convert latitude/longitude to grid ID using shapefile
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GridMapper:
    """
    Maps latitude/longitude coordinates to grid IDs
    Uses shapefile for spatial join
    """
    
    def __init__(self, shapefile_path: str = None):
        """Initialize with shapefile path"""
        if shapefile_path is None:
            # Default path relative to project root
            base_path = Path(__file__).parent.parent
            shapefile_path = base_path / "models" / "grids" / "city_grids_500m.shp"
        
        self.shapefile_path = shapefile_path
        self.grids_gdf = None
        self.is_loaded = False
        
    def load_grids(self) -> None:
        """Load the grid shapefile"""
        try:
            logger.info(f"Loading grid shapefile from: {self.shapefile_path}")
            self.grids_gdf = gpd.read_file(self.shapefile_path)
            self.is_loaded = True
            logger.info(f"Grid shapefile loaded successfully. Total grids: {len(self.grids_gdf)}")
            logger.info(f"CRS: {self.grids_gdf.crs}")
        except Exception as e:
            logger.error(f"Failed to load grid shapefile: {str(e)}")
            raise Exception(f"Grid shapefile loading failed: {str(e)}")
    
    def get_grid_id(self, latitude: float, longitude: float) -> Optional[int]:
        """
        Get grid ID for given lat/long coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Grid ID (int) if point falls within a grid, None otherwise
        """
        if not self.is_loaded:
            raise RuntimeError("Grids not loaded. Call load_grids() first.")
        
        try:
            # Create point geometry
            point = Point(longitude, latitude)  # Note: Point takes (x, y) = (lon, lat)
            
            # Create GeoDataFrame with the point
            point_gdf = gpd.GeoDataFrame(
                {'geometry': [point]}, 
                crs=self.grids_gdf.crs
            )
            
            # Perform spatial join to find which grid contains the point
            joined = gpd.sjoin(
                point_gdf, 
                self.grids_gdf, 
                how='left', 
                predicate='within'
            )
            
            # Get grid ID
            if not joined.empty and 'id' in joined.columns:
                grid_id = joined.iloc[0]['id']
                if pd.notna(grid_id):
                    logger.debug(f"Point ({latitude}, {longitude}) -> Grid ID: {grid_id}")
                    return int(grid_id)
            
            logger.warning(f"Point ({latitude}, {longitude}) does not fall within any grid")
            return None
            
        except Exception as e:
            logger.error(f"Error finding grid for coordinates ({latitude}, {longitude}): {str(e)}")
            raise Exception(f"Grid lookup failed: {str(e)}")
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """
        Validate if coordinates are within grid bounds
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            True if within bounds, False otherwise
        """
        if not self.is_loaded:
            logger.error("Cannot validate coordinates: Grids not loaded. Check if shapefile exists and is readable.")
            return False
        
        try:
            bounds = self.grids_gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            within_lon = bounds[0] <= longitude <= bounds[2]
            within_lat = bounds[1] <= latitude <= bounds[3]
            
            return within_lon and within_lat
        except Exception as e:
            logger.error(f"Error validating coordinates: {str(e)}")
            return False
    
    def get_grid_info(self) -> dict:
        """Get information about loaded grids"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        bounds = self.grids_gdf.total_bounds
        
        return {
            "status": "loaded",
            "total_grids": len(self.grids_gdf),
            "crs": str(self.grids_gdf.crs),
            "bounds": {
                "min_longitude": float(bounds[0]),
                "min_latitude": float(bounds[1]),
                "max_longitude": float(bounds[2]),
                "max_latitude": float(bounds[3])
            },
            "grid_id_column": "id"
        }

# Create global grid mapper instance
grid_mapper = GridMapper()
