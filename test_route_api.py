"""
Test script for Route Safety Prediction API
Tests the complete pipeline of the new /predict/route endpoint
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"

def test_route_prediction():
    """Test the route prediction endpoint"""
    
    # Request payload - simple route with 3 coordinates
    payload = {
        "coordinates": [
            {"lat": 37.7749, "lng": -122.4194},
            {"lat": 37.7750, "lng": -122.4195},
            {"lat": 37.7751, "lng": -122.4196}
        ],
        "datetime": "2026-02-20T14:30:00"
    }
    
    print("=" * 60)
    print("ROUTE SAFETY PREDICTION API TEST")
    print("=" * 60)
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        # Make request to route endpoint
        url = f"{BASE_URL}{API_PREFIX}/predict/route"
        print(f"\nCalling: POST {url}")
        
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS - Route Prediction Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            required_fields = [
                'route_safety_score',
                'route_safety_level',
                'worst_probability',
                'average_probability',
                'segment_count',
                'unsafe_segments'
            ]
            
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                print(f"\n⚠️ Missing fields: {missing_fields}")
            else:
                print(f"\n✅ All required fields present")
                
                # Print summary
                print(f"\n" + "=" * 60)
                print(f"ROUTE SAFETY SUMMARY")
                print(f"=" * 60)
                print(f"Average Safety Score: {result['average_probability']:.4f}")
                print(f"Worst Case Score:     {result['worst_probability']:.4f}")
                print(f"Overall Level:        {result['route_safety_level']}")
                print(f"Segments Evaluated:   {result['segment_count']}")
                print(f"Unsafe Segments:      {result['unsafe_segments']}")
                print(f"=" * 60)
        else:
            print(f"\n❌ ERROR - Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print(f"Make sure the API is running on {BASE_URL}")
        print(f"Run: python run.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


def test_route_with_longer_path():
    """Test with a longer route (more coordinates)"""
    
    print("\n\n" + "=" * 60)
    print("TEST 2: LONGER ROUTE (5 coordinates)")
    print("=" * 60)
    
    # A longer path with 5 coordinates
    payload = {
        "coordinates": [
            {"lat": 37.7749, "lng": -122.4194},
            {"lat": 37.7754, "lng": -122.4189},
            {"lat": 37.7759, "lng": -122.4184},
            {"lat": 37.7764, "lng": -122.4179},
            {"lat": 37.7769, "lng": -122.4174}
        ],
        "datetime": "2026-02-20T20:00:00"
    }
    
    print(f"\nRequest with {len(payload['coordinates'])} coordinates")
    print(f"DateTime: {payload['datetime']}")
    
    try:
        url = f"{BASE_URL}{API_PREFIX}/predict/route"
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS")
            print(f"Average Safety:  {result['average_probability']:.4f}")
            print(f"Worst Case:      {result['worst_probability']:.4f}")
            print(f"Safety Level:    {result['route_safety_level']}")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")


if __name__ == "__main__":
    test_route_prediction()
    test_route_with_longer_path()
    
    print("\n\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
