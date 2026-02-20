# Safety Score Model API

## 🎯 Overview

A FastAPI-based REST API for crime prediction using XGBoost. This system predicts whether a geographic grid cell will be unsafe during a specific 4-hour time window based on spatio-temporal features.

## 🏗️ Project Structure

```
safety_score_model/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic models for validation
│   └── predictor.py         # Model loading and prediction logic
├── models/
│   └── crime_grid_xgb_model.pkl  # Trained XGBoost model
├── requirements.txt         # Python dependencies
├── run.py                   # Server startup script
├── .env.example            # Environment variables example
└── README.md               # This file
```

## 📋 Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

## 🚀 Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

Copy `.env.example` to `.env` and modify if needed:
```bash
copy .env.example .env
```

## ▶️ Running the API

### Method 1: Using run.py (Recommended)

```bash
python run.py
```

### Method 2: Using uvicorn directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Using the app module

```bash
python -m app.main
```

The API will start at: `http://localhost:8000`

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔌 API Endpoints

### General Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/model/threshold` | POST | Update prediction threshold |

### Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Single prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |

## 📝 API Usage Examples

### Single Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "unsafe_probability": 0.67,
  "prediction": 1
}
```

### Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
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
      },
      {
        "index_right": 20,
        "time_bin": 2,
        "day_of_week": 1,
        "month": 5,
        "year": 2003,
        "is_weekend": 0,
        "lag_1": 1.0,
        "lag_6": 2.0,
        "lag_42": 3.0,
        "rolling_6_mean": 1.5,
        "rolling_42_mean": 2.2
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "unsafe_probability": 0.67,
      "prediction": 1
    },
    {
      "unsafe_probability": 0.85,
      "prediction": 1
    }
  ],
  "count": 2
}
```

## 🧪 Testing with Python

Create a test script `test_api.py`:

```python
import requests

# API Base URL
BASE_URL = "http://localhost:8000"

# Test single prediction
def test_single_prediction():
    url = f"{BASE_URL}/api/v1/predict"
    payload = {
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
    
    response = requests.post(url, json=payload)
    print("Single Prediction Response:")
    print(response.json())

# Run test
test_single_prediction()
```

## 🔧 Model Features

The model uses the following 11 features in **exact order**:

1. **index_right**: Grid cell identifier (int, ≥0)
2. **time_bin**: 4-hour time bin, 0-5 (int)
   - 0: 00:00-03:59
   - 1: 04:00-07:59
   - 2: 08:00-11:59
   - 3: 12:00-15:59
   - 4: 16:00-19:59
   - 5: 20:00-23:59
3. **day_of_week**: Day of week, 0-6 (int, 0=Monday)
4. **month**: Month, 1-12 (int)
5. **year**: Year (int)
6. **is_weekend**: Weekend indicator (int, 0 or 1)
7. **lag_1**: Crime count in previous 4-hour window (float, ≥0)
8. **lag_6**: Crime count 24 hours ago (float, ≥0)
9. **lag_42**: Crime count 7 days ago (float, ≥0)
10. **rolling_6_mean**: 24-hour rolling mean (float, ≥0)
11. **rolling_42_mean**: 7-day rolling mean (float, ≥0)

## ⚙️ Configuration

### Prediction Threshold

Default threshold: **0.3** (optimized for high recall)

Update threshold dynamically:
```bash
curl -X POST "http://localhost:8000/model/threshold?threshold=0.4"
```

### Model Performance

- **ROC-AUC**: 0.95
- **Recall (unsafe)**: ~1.00
- **Precision (unsafe)**: ~0.26
- **Approach**: Safety-first (prioritizes recall over precision)

## 🐛 Troubleshooting

### Model Not Loading

**Error:** `Model file not found`

**Solution:** Ensure `crime_grid_xgb_model.pkl` exists in `models/` folder

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:** Ensure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Change port in `.env` or run with custom port:
```bash
uvicorn app.main:app --port 8001
```

## 📊 Model Details

- **Algorithm**: XGBoost Binary Classifier
- **Library**: scikit-learn API (xgboost)
- **Version**: 3.2.0
- **Training Details**:
  - n_estimators: 1000
  - max_depth: 6
  - learning_rate: 0.05
  - scale_pos_weight: 18.28 (handles class imbalance)
  - eval_metric: AUC

## 🚢 Deployment

### Production Considerations

1. **Disable auto-reload:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Use production ASGI server:**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Update CORS settings** in `app/main.py` for security

4. **Set environment variables** appropriately

5. **Enable HTTPS** with reverse proxy (Nginx/Caddy)

## 📦 Dependencies

Key dependencies:
- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- XGBoost: ML model
- pandas: Data manipulation
- scikit-learn: ML utilities
- joblib: Model persistence

## 📄 License

[Add your license information]

## 👥 Contact

[Add your contact information]

---

**Built with FastAPI & XGBoost** 🚀
