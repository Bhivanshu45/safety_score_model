#!/usr/bin/env python
"""
Debug script to test grid loading and coordinate validation
Run: python debug_grids.py
"""

import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.geo_utils import grid_mapper
from app.datetime_utils import parse_datetime_string
from app.predictor import predictor

print("=" * 80)
print("GRID & MODEL DEBUG TEST")
print("=" * 80)

# Test 1: Model Loading
print("\n✓ TEST 1: MODEL LOADING")
print("-" * 80)
try:
    predictor.load_model()
    print(f"✅ Model loaded successfully")
    print(f"   - Status: {predictor.is_loaded}")
    print(f"   - Features: {predictor.feature_columns}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# Test 2: Grid Loading
print("\n✓ TEST 2: GRID SHAPEFILE LOADING")
print("-" * 80)
try:
    grid_mapper.load_grids()
    print(f"✅ Grids loaded successfully")
    print(f"   - Status: {grid_mapper.is_loaded}")
    print(f"   - Total grids: {len(grid_mapper.grids_gdf)}")
    print(f"   - CRS: {grid_mapper.grids_gdf.crs}")
    
    bounds = grid_mapper.grids_gdf.total_bounds
    print(f"   - Service Bounds:")
    print(f"     Latitude:  [{bounds[1]:.4f}, {bounds[3]:.4f}]")
    print(f"     Longitude: [{bounds[0]:.4f}, {bounds[2]:.4f}]")
except Exception as e:
    print(f"❌ Grid loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Coordinate Validation
print("\n✓ TEST 3: COORDINATE VALIDATION")
print("-" * 80)

test_coords = [
    (37.7749, -122.4194, "Local Test (SF)"),
    (37.7599, -122.4148, "Render Test (SF)"),
]

for lat, lon, label in test_coords:
    print(f"\n  Testing: {label}")
    print(f"  Coordinates: ({lat}, {lon})")
    
    try:
        # Check if valid
        is_valid = grid_mapper.validate_coordinates(lat, lon)
        print(f"  Valid in bounds: {is_valid}")
        
        # Try to get grid ID
        grid_id = grid_mapper.get_grid_id(lat, lon)
        if grid_id:
            print(f"  ✅ Grid ID found: {grid_id}")
        else:
            print(f"  ⚠️  No grid found for these coordinates")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Test 4: Full Prediction
print("\n✓ TEST 4: FULL LOCATION PREDICTION")
print("-" * 80)

try:
    # Parse datetime
    dt_str = "2026-02-20T14:30:00"
    dt = parse_datetime_string(dt_str)
    print(f"✅ Datetime parsed: {dt}")
    
    # Test coordinates
    lat, lon = 37.7749, -122.4194
    print(f"\nAttempting prediction for:")
    print(f"  Location: ({lat}, {lon})")
    print(f"  DateTime: {dt_str}")
    
    result = predictor.predict_from_location(
        latitude=lat,
        longitude=lon,
        dt=dt,
        grid_mapper=grid_mapper
    )
    
    print(f"\n✅ PREDICTION SUCCESSFUL!")
    print(f"   - Unsafe Probability: {result.unsafe_probability:.4f}")
    print(f"   - Prediction: {result.prediction}")
    
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG TEST COMPLETE")
print("=" * 80)
