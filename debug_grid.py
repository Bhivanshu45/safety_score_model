"""
Diagnostic script to debug grid loading and coordinate validation issues
"""

import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.geo_utils import grid_mapper
from app.datetime_utils import parse_datetime_string, extract_temporal_features
from app.predictor import predictor

print("="*60)
print("🔍 DIAGNOSTIC SCRIPT - Grid & Coordinate Validation")
print("="*60)

# Test 1: Check grid loading
print("\n1️⃣ GRID LOADING TEST")
print("-" * 60)
try:
    grid_mapper.load_grids()
    print(f"✅ Grid loaded successfully")
    print(f"   Total grids: {len(grid_mapper.grids_gdf)}")
    print(f"   CRS: {grid_mapper.grids_gdf.crs}")
    
    # Get bounds
    bounds = grid_mapper.grids_gdf.total_bounds
    print(f"   Bounds (minx, miny, maxx, maxy):")
    print(f"     Min Longitude: {bounds[0]:.4f}")
    print(f"     Min Latitude: {bounds[1]:.4f}")
    print(f"     Max Longitude: {bounds[2]:.4f}")
    print(f"     Max Latitude: {bounds[3]:.4f}")
    
except Exception as e:
    print(f"❌ Grid loading failed: {str(e)}")
    sys.exit(1)

# Test 2: Check coordinate validation for the test coordinates
print("\n2️⃣ COORDINATE VALIDATION TEST")
print("-" * 60)

test_coords = [
    (37.7749, -122.4194, "San Francisco (Local test)"),
    (37.7599, -122.4148, "San Francisco (Render test)"),
]

for lat, lon, description in test_coords:
    print(f"\nTesting: {description}")
    print(f"  Coordinates: ({lat}, {lon})")
    
    # Check bounds
    bounds = grid_mapper.grids_gdf.total_bounds
    within_lon = bounds[0] <= lon <= bounds[2]
    within_lat = bounds[1] <= lat <= bounds[3]
    
    print(f"  Within longitude bounds? {within_lon}")
    if not within_lon:
        print(f"    (Expected: {bounds[0]:.4f} <= {lon} <= {bounds[2]:.4f})")
    
    print(f"  Within latitude bounds? {within_lat}")
    if not within_lat:
        print(f"    (Expected: {bounds[1]:.4f} <= {lat} <= {bounds[3]:.4f})")
    
    # Try to get grid ID
    try:
        grid_id = grid_mapper.get_grid_id(lat, lon)
        if grid_id is not None:
            print(f"  ✅ Grid ID found: {grid_id}")
        else:
            print(f"  ❌ Grid ID not found (coordinates outside all grids)")
    except Exception as e:
        print(f"  ❌ Error getting grid ID: {str(e)}")

# Test 3: Check model loading
print("\n3️⃣ MODEL LOADING TEST")
print("-" * 60)
try:
    predictor.load_model()
    print(f"✅ Model loaded successfully")
    print(f"   Features expected: {len(predictor.feature_columns)}")
    print(f"   Feature columns: {predictor.feature_columns}")
    print(f"   Threshold: {predictor.threshold}")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    sys.exit(1)

# Test 4: Full prediction test
print("\n4️⃣ FULL PREDICTION TEST")
print("-" * 60)

test_request = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "datetime": "2026-02-20T14:30:00"
}

print(f"Request: {test_request}")

try:
    dt = parse_datetime_string(test_request["datetime"])
    temporal_features = extract_temporal_features(dt)
    print(f"\nTemporal Features:")
    for key, value in temporal_features.items():
        print(f"  {key}: {value}")
    
    # Check coordinates
    is_valid = grid_mapper.validate_coordinates(test_request["latitude"], test_request["longitude"])
    print(f"\nCoordinates valid: {is_valid}")
    
    if not is_valid:
        print("❌ Coordinates are outside grid bounds!")
        print("This is why prediction fails on deployment.")
    else:
        # Try full prediction
        result = predictor.predict_from_location(
            latitude=test_request["latitude"],
            longitude=test_request["longitude"],
            dt=dt,
            grid_mapper=grid_mapper
        )
        print(f"\n✅ Prediction successful:")
        print(f"   Unsafe Probability: {result.unsafe_probability}")
        print(f"   Prediction: {result.prediction}")
        
except Exception as e:
    print(f"❌ Prediction failed: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 5: Check grid bounds for all grids
print("\n5️⃣ GRID BOUNDS ANALYSIS")
print("-" * 60)
try:
    gdf = grid_mapper.grids_gdf
    
    # Get all columns
    print(f"Available columns in grid: {list(gdf.columns)}")
    
    # Check ID column
    if 'id' in gdf.columns:
        print(f"\n✅ 'id' column found")
        print(f"   Sample grid IDs: {list(gdf['id'].head(5))}")
    else:
        print(f"\n❌ 'id' column NOT found!")
        print(f"   Available columns: {list(gdf.columns)}")
    
    # Print some sample geometries
    print(f"\nSample grid bounds (first 3):")
    for idx, (i, row) in enumerate(gdf.head(3).iterrows()):
        bounds = row.geometry.bounds
        print(f"  Grid {row.get('id', 'N/A')}: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")
        if idx >= 2:
            break
    
except Exception as e:
    print(f"❌ Error analyzing grids: {str(e)}")

print("\n" + "="*60)
print("✅ Diagnostic complete!")
print("="*60)
