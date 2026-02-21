"""
Direct Route Prediction Test (Offline)
Test the route prediction pipeline without running the API server
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.predictor import predictor
from app.geo_utils import grid_mapper
from app.datetime_utils import parse_datetime_string


def test_route_prediction_offline():
    """Test route prediction using model directly"""
    
    print("=" * 70)
    print("OFFLINE ROUTE PREDICTION TEST")
    print("=" * 70)
    
    try:
        # Load model and grids
        print("\n[1/4] Loading model...")
        predictor.load_model()
        print("      ✅ Model loaded")
        
        print("[2/4] Loading grid mapper...")
        grid_mapper.load_grids()
        print("      ✅ Grid mapper loaded")
        
        # Test coordinates (example route)
        coordinates = [
            (37.7749, -122.4194),  # Start
            (37.7750, -122.4195),
            (37.7751, -122.4196)   # End
        ]
        
        # Parse datetime
        print("\n[3/4] Parsing datetime...")
        dt = parse_datetime_string("2026-02-20T14:30:00")
        print(f"      ✅ DateTime: {dt}")
        
        # Run prediction
        print("\n[4/4] Running route prediction...")
        print(f"      Coordinates: {len(coordinates)} points")
        print(f"      Route: {coordinates[0]} → {coordinates[-1]}")
        
        result = predictor.predict_route(coordinates, dt, grid_mapper)
        
        print("\n" + "=" * 70)
        print("ROUTE PREDICTION RESULTS")
        print("=" * 70)
        print(f"\n✅ Route Safety Score (Average):  {result['route_safety_score']:.4f}")
        print(f"✅ Route Safety Level:             {result['route_safety_level']}")
        print(f"\nDetailed Metrics:")
        print(f"  • Average Probability:            {result['average_probability']:.4f}")
        print(f"  • Worst Case Probability:         {result['worst_probability']:.4f}")
        print(f"  • Total Segments:                 {result['segment_count']}")
        print(f"  • Unsafe Segments:                {result['unsafe_segments']}")
        
        # Map score to risk level
        print(f"\nSafety Level Breakdown:")
        if result['average_probability'] < 0.33:
            print(f"  Low Risk:     Score < 0.33")
        elif result['average_probability'] < 0.67:
            print(f"  Medium Risk:  0.33 ≤ Score < 0.67")
        else:
            print(f"  High Risk:    Score ≥ 0.67")
        
        print("\n" + "=" * 70)
        print("TEST PASSED ✅")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_route_prediction_offline()
    sys.exit(0 if success else 1)
