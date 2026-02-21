"""
Geospatial Utilities
Convert latitude/longitude to grid ID using shapefile (Pure Python, no GDAL)
"""

import shapefile
from shapely.geometry import Point, shape
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GridMapper:
    """
    Maps latitude/longitude coordinates to grid IDs
    Uses pure Python shapefile reading (no GDAL/Fiona required)
    """
    
    def __init__(self, shapefile_path: str = None):
        """Initialize with shapefile path"""
        if shapefile_path is None:
            # Default path relative to project root
            base_path = Path(__file__).parent.parent
            shapefile_path = base_path / "models" / "grids" / "city_grids_500m.shp"
        
        self.shapefile_path = str(shapefile_path)
        self.shapes_data = []  # List of (geometry, id) tuples
        self.is_loaded = False
        self.bounds = None
        
    def load_grids(self) -> None:
        """Load the grid shapefile using pure Python"""
        try:
            logger.info(f"Loading grid shapefile from: {self.shapefile_path}")
            
            # Read shapefile using pyshp (pure Python)
            sf = shapefile.Reader(self.shapefile_path)
            
            logger.info(f"Shapefile loaded - {len(sf.shapes())} shapes found")
            
            # Extract shapes and IDs
            self.shapes_data = []
            for sr in sf.shapeRecords():
                geom = shape(sr.shape.__geo_interface__)
                grid_id = sr.record[0]  # First field is 'id'
                self.shapes_data.append((geom, grid_id))
            
            # Calculate bounds
            all_bounds = [geom.bounds for geom, _ in self.shapes_data]
            if all_bounds:
                min_x = min(b[0] for b in all_bounds)
                min_y = min(b[1] for b in all_bounds)
                max_x = max(b[2] for b in all_bounds)
                max_y = max(b[3] for b in all_bounds)
                self.bounds = (min_x, min_y, max_x, max_y)
            
            self.is_loaded = True
            logger.info(f"Grid shapefile loaded successfully. Total grids: {len(self.shapes_data)}")
            logger.info(f"Bounds: Lon [{self.bounds[0]:.4f}, {self.bounds[2]:.4f}], "
                       f"Lat [{self.bounds[1]:.4f}, {self.bounds[3]:.4f}]")
            
        except Exception as e:
            logger.error(f"Failed to load grid shapefile: {str(e)}")
            raise Exception(f"Grid shapefile loading failed: {str(e)}")
    
    def get_grid_id(self, latitude: float, longitude: float) -> Optional[int]:
        """
        Get grid ID for given lat/long coordinates (Pure Python)
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Grid ID (int) if point falls within a grid, None otherwise
        """
        if not self.is_loaded:
            raise RuntimeError("Grids not loaded. Call load_grids() first.")
        
        try:
            # Create point geometry (Point takes x, y = lon, lat)
            point = Point(longitude, latitude)
            
            # Find which grid contains this point
            for geom, grid_id in self.shapes_data:
                if geom.contains(point):
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
        if not self.is_loaded or not self.bounds:
            return False
        
        min_x, min_y, max_x, max_y = self.bounds
        
        within_lon = min_x <= longitude <= max_x
        within_lat = min_y <= latitude <= max_y
        
        return within_lon and within_lat
    
    def get_grid_info(self) -> dict:
        """Get information about loaded grids"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "total_grids": len(self.shapes_data),
            "bounds": {
                "min_longitude": float(self.bounds[0]),
                "min_latitude": float(self.bounds[1]),
                "max_longitude": float(self.bounds[2]),
                "max_latitude": float(self.bounds[3])
            },
            "grid_id_column": "id"
        }
    
    def get_all_grids_geometries(self) -> list:
        """
        Get all grid geometries in GeoJSON format
        
        Returns:
            List of tuples (grid_id, geometry_dict) where geometry is GeoJSON polygon
        """
        if not self.is_loaded:
            raise RuntimeError("Grids not loaded. Call load_grids() first.")
        
        grids_geojson = []
        
        for geom, grid_id in self.shapes_data:
            # Convert shapely geometry to GeoJSON-like dict
            # Use __geo_interface__ for GeoJSON representation
            geojson_geom = {
                "type": geom.geom_type,
                "coordinates": list(geom.exterior.coords) if geom.geom_type == "Polygon" else None
            }
            
            # Convert coordinates from (lon, lat) tuples to [[lon, lat], ...] lists
            if geojson_geom["type"] == "Polygon":
                geojson_geom["coordinates"] = [[[lon, lat] for lon, lat in geojson_geom["coordinates"]]]
            
            grids_geojson.append((int(grid_id), geojson_geom))
        
        logger.info(f"Generated GeoJSON geometries for {len(grids_geojson)} grids")
        return grids_geojson


# Create global grid mapper instance
grid_mapper = GridMapper()


# Create global grid mapper instance
grid_mapper = GridMapper()
