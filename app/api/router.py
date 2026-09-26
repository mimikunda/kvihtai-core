"""Aggregates every route module into a single router.

The networking layer lives under `app.api` only.
"""

from fastapi import APIRouter

from app.api.routes import health, sets, station

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(sets.router)
api_router.include_router(station.router)
