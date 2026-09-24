"""FastAPI application factory."""

from fastapi import FastAPI

from app.api.router import api_router
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(api_router)
    return app


app = create_app()
