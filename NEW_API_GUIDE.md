# 🎯 NEW SIMPLIFIED API - READY TO USE!

## ✅ What's New:

### **OLD API (Complex):** ❌
```json
{
  "index_right": 15,
  "time_bin": 3,
  "day_of_week": 5,
  "month": 2,
  "year": 2003,
  "is_weekend": 1,
  "lag_1": 0.0,
  "lag_6": 0.0,
  "lag_42": 0.0,
  "rolling_6_mean": 0.0,
  "rolling_42_mean": 0.0
}
```
**Problem:** Frontend can't provide these values!

---

### **NEW API (Simplified):** ✅
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "datetime": "2026-02-20T14:30:00"
}
```
**Perfect:** Frontend only needs current location + time!

---

## 📡 API Endpoints

|  | Endpoint | Input | Best For |
|--|----------|-------|----------|
| 🆕 | `/api/v1/predict/location` | lat, long, datetime | **Frontend Apps** |
| 📊 | `/api/v1/predict` | All 11 features | Testing/Advanced |
| 📦 | `/api/v1/predict/batch` | Multiple predictions | Batch processing |

---

## 🚀 NEW API USAGE

### **Endpoint:**
```
POST http://localhost:8000/api/v1/predict/location
```

### **Request:**
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "datetime": "2026-02-20T14:30:00"
}
```

### **Response:**
```json
{
  "unsafe_probability": 0.67,
  "prediction": 1,
  "safety_level": "High Risk",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "grid_id": 15
  },
  "time_context": {
    "datetime": "2026-02-20T14:30:00",
    "time_period": "12:00-15:59 (Afternoon)",
    "day": "Thursday",
    "is_weekend": false
  },
  "details": {
    "time_bin": 3,
    "day_of_week": 3,
    "month": 2,
    "year": 2026
  }
}
```

---

## 🔥 What the API Does Automatically:

1. **Lat/Long → Grid ID** (using shapefile)
2. **DateTime → Temporal Features** (time_bin, day_of_week, etc.)
3. **Historical Features** (defaults to 0 for now)
4. **Model Prediction**
5. **Enhanced Response** (with safety level and context)

---

## 🧪 HOW TO TEST:

### **Option 1: Python Script**
```bash
python test_location_api.py
```

### **Option 2: Browser (Interactive)**
```
http://localhost:8000/docs
```
- Find `/api/v1/predict/location`
- Click "Try it out"
- Use the payload above
- Click "Execute"

### **Option 3: curl**
```bash
curl -X POST http://localhost:8000/api/v1/predict/location ^
  -H "Content-Type: application/json" ^
  -d "{\"latitude\":37.7749,\"longitude\":-122.4194,\"datetime\":\"2026-02-20T14:30:00\"}"
```

---

## 📍 Valid Locations:

Your shapefile covers San Francisco:
- **Latitude:** 37.70 to 37.82
- **Longitude:** -122.52 to -122.36

Coordinates outside this area will return an error.

---

## 🎨 Safety Levels:

| Probability | Level | Meaning |
|-------------|-------|---------|
| < 0.3 | Low Risk | ✅ Safe |
| 0.3 - 0.5 | Moderate Risk | ⚠️ Caution |
| 0.5 - 0.7 | High Risk | 🔶 Be Careful |
| > 0.7 | Very High Risk | 🔴 Dangerous |

---

## 🔧 Technical Details:

### Files Added:
- `app/geo_utils.py` - Lat/long → Grid conversion
- `app/datetime_utils.py` - DateTime → Temporal features
- Updated `app/models.py` - New request/response models
- Updated `app/main.py` - New endpoint
- Updated `app/predictor.py` - Location prediction method

### Dependencies Added:
- `geopandas` - Geospatial operations
- `shapely` - Geometry operations

---

## ✅ READY TO USE!

The new API is **production-ready** and **frontend-friendly**!

Just restart the server and test with `test_location_api.py`! 🚀
