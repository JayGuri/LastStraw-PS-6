"""Main FastAPI application for HackX backend."""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mongo.routes import router as auth_router
from mongo.client import db, ensure_indexes

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackX Backend",
    description="MongoDB + OAuth authentication service",
    version="1.0.0",
)

# Configure CORS: localhost for dev, production from env (e.g. Vercel frontend URL)
_cors_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
# Add production frontend URL so OAuth redirect and API calls work on Vercel
_frontend_url = os.getenv("FRONTEND_URL", "").strip()
if _frontend_url and _frontend_url not in _cors_origins:
    _cors_origins.append(_frontend_url)
# Optional: comma-separated list of extra allowed origins
_extra = os.getenv("CORS_ORIGINS", "").strip()
if _extra:
    _cors_origins.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])


@app.on_event("startup")
def startup():
    """Run MongoDB index creation after app is loaded (non-blocking)."""
    ensure_indexes()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "HackX backend is running"}


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "HackX API", "docs": "/docs"}
