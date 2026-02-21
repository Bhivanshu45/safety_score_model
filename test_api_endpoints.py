"""
Test API endpoints when server is running
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing Safety Score Model APIs")
print("=" * 60)

# Test 1: Health check
print("\n[1/4] Testing Health Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health Status: {data['status']}")
        print(f"   Model Loaded: {data['model_loaded']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Health endpoint error: {e}")

# Test 2: Location prediction
print("\n[2/4] Testing Location Prediction Endpoint...")
try:
    location_payload = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "datetime": "2026-02-20T14:30:00"
    }
    response = requests.post(
        f"{BASE_URL}/api/predict/location",
        json=location_payload,
        timeout=5
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Location Prediction Success")
        print(f"   Unsafe Probability: {data['unsafe_probability']:.4f}")
        print(f"   Safety Level: {data['safety_level']}")
    else:
        print(f"❌ Location prediction failed: {response.status_code}")
except Exception as e:
    print(f"❌ Location endpoint error: {e}")

# Test 3: Route prediction
print("\n[3/4] Testing Route Prediction Endpoint...")
try:
    route_payload = {
        "coordinates": [
            {"lat": 37.7749, "lng": -122.4194},
            {"lat": 37.7750, "lng": -122.4195},
            {"lat": 37.7751, "lng": -122.4196}
        ],
        "datetime": "2026-02-20T14:30:00"
    }
    response = requests.post(
        f"{BASE_URL}/api/predict/route",
        json=route_payload,
        timeout=5
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Route Prediction Success")
        print(f"   Route Safety Score: {data['route_safety_score']:.4f}")
        print(f"   Safety Level: {data['route_safety_level']}")
        print(f"   Avg Probability: {data['average_probability']:.4f}")
        print(f"   Worst Probability: {data['worst_probability']:.4f}")
        print(f"   Segments: {data['segment_count']} | Unsafe: {data['unsafe_segments']}")
    else:
        print(f"❌ Route prediction failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Route endpoint error: {e}")

# Test 4: Heatmap endpoint
print("\n[4/4] Testing Heatmap Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/heatmap", timeout=10)
    if response.status_code == 200:
        data = response.json()
        stats = data['statistics']
        print(f"✅ Heatmap Retrieved Successfully")
        print(f"   Total Grids: {stats['total_grids']}")
        print(f"   Safe: {stats['safe_count']} | Low Risk: {stats['low_risk_count']}")
        print(f"   Medium Risk: {stats['medium_risk_count']} | High Risk: {stats['high_risk_count']}")
        print(f"   Avg Unsafe Prob: {stats['average_unsafe_probability']:.4f}")
        print(f"   Response Size: {len(json.dumps(data)) / 1024:.2f} KB")
    else:
        print(f"❌ Heatmap failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Heatmap endpoint error: {e}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)
