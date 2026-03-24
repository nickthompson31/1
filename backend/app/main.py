"""
Portrait STL Generator — FastAPI Application

A tool that converts portrait photographs into carve-ready
bas-relief STL files using AI depth estimation and face-aware
processing to produce natural, non-creepy results.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="Portrait STL Generator",
    description=(
        "Convert portrait photos to beautiful, carve-ready bas-relief STL files. "
        "Uses AI depth estimation with face-aware processing to produce natural results."
    ),
    version="0.1.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "Portrait STL Generator",
        "version": "0.1.0",
        "docs": "/docs",
    }
