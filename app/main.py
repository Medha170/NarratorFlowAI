# app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.model_service import load_emotion_model, emotion_classifier
from api.v1.endpoints.narrative import router as narrative_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for loading and unloading models cleanly on startup/shutdown."""
    load_emotion_model()
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