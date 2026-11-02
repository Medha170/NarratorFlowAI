# app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.model_service import emotion_classifier
from app.api.v1.endpoints.narrative import router as narrative_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager. Model loading is now deferred to first request (lazy loading)."""
    print("FastAPI server successfully (Model will lazy-load on first request).")
    yield
    print("Shutting down application...")


app = FastAPI(
    title="Narrative Flow AI",
    description="High-performance async engine for deep emotional arc and narrative analysis.",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(narrative_router)


@app.get("/")
async def health_check():
    """Health check endpoint for production load balancers and container probes."""
    return {
        "status": "healthy",
        "model_loaded": emotion_classifier is not None
    }