"""FastAPI application factory.

The station, the camera and everything behind it, starts with the app and
stops with it. KVIHTAI_CAMERA=none, the default, runs the API alone.
The web app, if it has been built, is served at /.
"""

import asyncio
import logging
import time
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

    async def beat():
        # the station tells systemd it is alive only while this runs, see app.station
        while True:
            station.loop_beat = time.monotonic()
            await asyncio.sleep(2)

    beating = asyncio.create_task(beat())
    try:
        yield
    finally:
        beating.cancel()
        station.stop()


class WebFiles(StaticFiles):
    """The web app. Its index is asked for again every time, so that a phone
    picks up a new version at once; the files it names have their hash in the
    name and never change."""

    def file_response(self, full_path, *args, **kwargs):
        response = super().file_response(full_path, *args, **kwargs)
        immutable = "/assets/" in str(full_path).replace("\\", "/")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return response


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.include_router(api_router)
    if settings.web_dir.is_dir():
        app.mount("/", WebFiles(directory=settings.web_dir, html=True), name="web")
    return app


app = create_app()
