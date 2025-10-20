from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router

app = FastAPI(
    title="NSE News Sentiment Analysis API",
    description="Backend service for autocomplete, sentiment pipeline, and recent search endpoints",
    version="1.0.0"
)

# 🌐 Enable CORS (important for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 or set ["http://localhost:3000"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📡 Include all routes under /api
app.include_router(api_router, prefix="/api")


# 🩺 Optional health check or root route
@app.get("/")
def read_root():
    return {
        "message": " Backend is running",
        "docs": "/docs",
        "redoc": "/redoc"
    }
@app.get("/health")
def health():
    return {"status": "ok"}



