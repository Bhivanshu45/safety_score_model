"""
Run script for Safety Score Model API
Simple script to start the FastAPI server
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"📚 API Docs: http://localhost:{settings.port}/docs")
    print(f"📖 ReDoc: http://localhost:{settings.port}/redoc")
    print("-" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )
