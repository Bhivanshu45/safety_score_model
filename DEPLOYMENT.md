# Safety Score Model API - Deployment Guide

## Quick Deploy to Render

### 1. Push to GitHub
```bash
git add .
git commit -m "Production ready: Fixed PORT for Render deployment"
git push origin main
```

### 2. Deploy on Render

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect repository: `Bhivanshu45/safety_score_model`
4. Configure:
   - **Name**: `safety-score-api` (or your choice)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install --only-binary=:all: -r requirements.txt || pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

5. **Environment Variables**: **NONE NEEDED** ✅
   - No database credentials
   - No API keys  
   - Model files included in repo

6. **Advanced Settings** (Optional but recommended):
   - **Python Version**: Uses `runtime.txt` (Python 3.11.9) automatically

7. Click **"Create Web Service"**

### 3. Wait for Deployment
- First build: ~5-10 minutes
- Render will automatically use PORT from environment
- Check logs for "Application startup complete"

### 4. Test Your API
Your API will be live at: `https://your-service-name.onrender.com`

Test endpoints:
- **Docs**: `https://your-service-name.onrender.com/docs`
- **Health**: `https://your-service-name.onrender.com/health`
- **Predict**: `POST https://your-service-name.onrender.com/api/v1/predict/location`

### Sample Test Payload
```json
{
  "latitude": 37.7599,
  "longitude": -122.4148,
  "datetime": "2026-02-20T22:00:00"
}
```

## Local Development
```bash
python run.py
# Runs on http://localhost:8000
```

## Notes
- Free tier may sleep after 15 min inactivity
- First request after sleep takes ~30 seconds
- Model size: 4.2 MB (within limits)
- No paid services required

## Troubleshooting

### Build Error: pydantic-core or Rust compilation
**Fixed**: Updated `requirements.txt` with Render-compatible versions (pydantic 2.9.2)

### Build taking too long
- First build: 5-10 minutes (installing geopandas dependencies)
- Subsequent builds: ~2-3 minutes (cached)

### Service not responding
- Check Render logs for "Application startup complete" message
- Verify model file loaded successfully
- Free tier services sleep after inactivity - first request wakes it up (~30s)

### Need help?
Check Render logs in dashboard for detailed error messages
