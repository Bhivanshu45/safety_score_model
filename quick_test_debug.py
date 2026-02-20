"""Quick test to see model behavior"""
import requests
import json

payload = {
    "latitude": 37.7599,
    "longitude": -122.4148,
    "datetime": "2026-02-20T22:00:00"
}

print("Testing location prediction...")
response = requests.post(
    "http://localhost:8000/api/v1/predict/location",
    json=payload
)

result = response.json()
print(json.dumps(result, indent=2))
