"""FastAPI application factory.

The station, the camera and everything behind it, starts with the app and
stops with it. KVIHTAI_CAMERA=none, the default, runs the API alone.
The web app, if it has been built, is served at /.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.db import repository
    from app.station import Station

    station = Station(settings, store=repository)
    app.state.station = station
    station.start()
    try:
        yield
    finally:
        station.stop()


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.include_router(api_router)
    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app


app = create_app()
