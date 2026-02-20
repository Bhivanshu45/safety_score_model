"""
Test Location-Based Prediction API
Tests the simplified /predict/location endpoint
"""
import requests
import json
from datetime import datetime

# API URL
API_URL = "http://localhost:8000"

print("="*70)
print("🧪 Testing Location-Based Prediction API")
print("="*70)

# Test Case 1: Valid San Francisco Location
print("\n1️⃣ Test Case 1: Downtown San Francisco")
print("-" * 70)

payload_1 = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "datetime": "2026-02-20T14:30:00"
}

print(f"📤 Request:")
print(json.dumps(payload_1, indent=2))

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict/location",
        json=payload_1,
        timeout=10
    )
    
    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        
        print(f"\n✅ Summary:")
        print(f"   Safety Level: {result['safety_level']}")
        print(f"   Risk Probability: {result['unsafe_probability']:.1%}")
        print(f"   Prediction: {'🔴 UNSAFE' if result['prediction'] == 1 else '🟢 SAFE'}")
        print(f"   Grid ID: {result['location']['grid_id']}")
        print(f"   Time Period: {result['time_context']['time_period']}")
        print(f"   Day: {result['time_context']['day']}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 2: Evening Time
print("\n\n2️⃣ Test Case 2: Same Location - Evening Time")
print("-" * 70)

payload_2 = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "datetime": "2026-02-20T22:30:00"  # Late night
}

print(f"📤 Request:")
print(json.dumps(payload_2, indent=2))

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict/location",
        json=payload_2,
        timeout=10
    )
    
    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(f"   Safety Level: {result['safety_level']}")
        print(f"   Risk Probability: {result['unsafe_probability']:.1%}")
        print(f"   Prediction: {'🔴 UNSAFE' if result['prediction'] == 1 else '🟢 SAFE'}")
        print(f"   Time Period: {result['time_context']['time_period']}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 3: Weekend
print("\n\n3️⃣ Test Case 3: Weekend Night")
print("-" * 70)

payload_3 = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "datetime": "2026-02-21T23:00:00"  # Saturday night
}

print(f"📤 Request:")
print(json.dumps(payload_3, indent=2))

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict/location",
        json=payload_3,
        timeout=10
    )
    
    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(f"   Safety Level: {result['safety_level']}")
        print(f"   Risk Probability: {result['unsafe_probability']:.1%}")
        print(f"   Prediction: {'🔴 UNSAFE' if result['prediction'] == 1 else '🟢 SAFE'}")
        print(f"   Day: {result['time_context']['day']}")
        print(f"   Is Weekend: {result['time_context']['is_weekend']}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 4: Different Location
print("\n\n4️⃣ Test Case 4: Different Location")
print("-" * 70)

payload_4 = {
    "latitude": 37.7849,
    "longitude": -122.4094,
    "datetime": "2026-02-20T08:30:00"  # Morning
}

print(f"📤 Request:")
print(json.dumps(payload_4, indent=2))

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict/location",
        json=payload_4,
        timeout=10
    )
    
    print(f"\n📥 Response (Status: {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(f"   Safety Level: {result['safety_level']}")
        print(f"   Risk Probability: {result['unsafe_probability']:.1%}")
        print(f"   Prediction: {'🔴 UNSAFE' if result['prediction'] == 1 else '🟢 SAFE'}")
        print(f"   Grid ID: {result['location']['grid_id']}")
        print(f"   Time Period: {result['time_context']['time_period']}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test Case 5: Invalid Location (Outside bounds)
print("\n\n5️⃣ Test Case 5: Invalid Location (Should Fail)")
print("-" * 70)

payload_5 = {
    "latitude": 40.7128,  # New York coordinates
    "longitude": -74.0060,
    "datetime": "2026-02-20T14:30:00"
}

print(f"📤 Request:")
print(json.dumps(payload_5, indent=2))

try:
    response = requests.post(
        f"{API_URL}/api/v1/predict/location",
        json=payload_5,
        timeout=10
    )
    
    print(f"\n📥 Response (Status: {response.status_code}):")
    print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("✅ All tests completed!")
print("="*70)

print("\n📋 API SUMMARY:")
print(f"   Endpoint: POST {API_URL}/api/v1/predict/location")
print(f"   Input: latitude, longitude, datetime")
print(f"   Output: unsafe_probability, prediction, safety_level, context")
print("="*70)
