"""The station: what the camera sees and does, and its settings."""

import asyncio

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(tags=["station"])

LIVE_HZ = 25            # the camera runs faster; a phone does not need more


class CameraChange(BaseModel):
    quarter_turns: int | None = None
    exposure_us: int | None = None
    gain: float | None = None
    auto_gain: bool | None = None


class ClockReading(BaseModel):
    epoch_ms: float


def _station(request: Request):
    station = getattr(request.app.state, "station", None)
    if station is None:
        raise HTTPException(status_code=503, detail="The station is not running")
    return station


@router.get("/api/v1/status")
def status(request: Request) -> dict:
    return _station(request).status()


@router.get("/api/v1/camera")
def camera(request: Request) -> dict:
    return _station(request).camera_state()


@router.put("/api/v1/camera")
def change_camera(change: CameraChange, request: Request) -> dict:
    return _station(request).update_camera(**change.model_dump(exclude_none=True))


@router.post("/api/v1/camera/focus")
def focus(request: Request) -> dict:
    if not _station(request).autofocus():
        raise HTTPException(status_code=409, detail="This camera cannot focus")
    return {"focusing": True}


@router.get("/api/v1/camera/preview.jpg")
def preview(request: Request, width: int = 480):
    jpeg = _station(request).preview_jpeg(width=max(64, min(width, 1920)))
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No picture yet")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/v1/clock")
def clock(reading: ClockReading, request: Request) -> dict:
    return _station(request).set_clock(reading.epoch_ms)


@router.get("/api/v1/log")
def recent_log(request: Request) -> list[str]:
    return list(_station(request).lines)


@router.get("/api/v1/recordings")
def recordings(request: Request) -> dict:
    station = _station(request)
    rec = station.recorder
    segments = rec.segments() if rec is not None else []
    return {
        "enabled": rec is not None,
        "segments": [{k: s.get(k) for k in ("name", "started_at", "bytes", "duration_s", "frames", "recording")}
                     for s in segments],
        "total_bytes": sum(s["bytes"] for s in segments),
    }


@router.websocket("/ws/live")
async def live(websocket: WebSocket):
    """The bar's position and speed while a set goes on; a heartbeat while idle."""
    station = getattr(websocket.app.state, "station", None)
    await websocket.accept()
    if station is None:
        await websocket.close(code=1013)
        return
    last = None
    quiet = 0
    try:
        while True:
            payload = station.live_payload()
            key = (payload["lift_state"], payload["bar_center"]["x_mm"], payload["bar_center"]["y_mm"])
            quiet += 1
            if key != last or quiet >= LIVE_HZ:
                await websocket.send_json(payload)
                last, quiet = key, 0
            await asyncio.sleep(1 / LIVE_HZ)
    except (WebSocketDisconnect, RuntimeError):
        pass
