"""The computer and the station's upkeep: power, problems, logs and a bundle of
everything for a bug report."""

import io
import json
import os
import socket
import zipfile
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app import logbook, system
from app.api.routes.sets import brief

router = APIRouter(tags=["system"])

MAX_LINES = 5000

# what goes into the diagnostics bundle from the journal: (source, boot, lines, problems only)
JOURNALS = {
    "station-this-boot": ("station", 0, 20000, False),
    "station-previous-boot": ("station", -1, 5000, False),
    "kernel-this-boot": ("kernel", 0, 5000, False),
    "kernel-previous-boot": ("kernel", -1, 2000, False),
    "system-problems-this-boot": ("system", 0, 2000, True),
}


class PowerRequest(BaseModel):
    action: Literal["restart", "reboot", "shutdown"]


def _station(request: Request):
    station = getattr(request.app.state, "station", None)
    if station is None:
        raise HTTPException(status_code=503, detail="The station is not running")
    return station


@router.get("/api/v1/system")
def system_info(request: Request) -> dict:
    station = _station(request)
    out = system.info(station.data_dir)
    out["power"] = station.power_abilities()
    out["service"] = system.under_systemd()
    out["storage"] = {
        "recordings_bytes": sum(s["bytes"] for s in station.recordings()),
        "sets_bytes": system.directory_bytes(station.sets_dir),
        "logs_bytes": system.directory_bytes(station.log_dir),
    }
    return out


@router.post("/api/v1/system/power", status_code=202)
def power(req: PowerRequest, request: Request) -> dict:
    accepted, why = _station(request).power(req.action)
    if not accepted:
        raise HTTPException(status_code=409, detail=why)
    return {"action": req.action}


@router.get("/api/v1/incidents")
def incidents(request: Request) -> list[dict]:
    return _station(request).incidents


@router.delete("/api/v1/incidents")
def clear_incidents(request: Request) -> dict:
    _station(request).clear_incidents()
    return {"cleared": True}


@router.get("/api/v1/system/log")
def station_log(request: Request, lines: int = 400, problems: bool = False) -> dict:
    """The station's own log from the card, across restarts; a record may hold several lines."""
    station = _station(request)
    return {"records": logbook.read(station.log_dir, max(1, min(lines, MAX_LINES)), problems)}


@router.get("/api/v1/system/journal")
def journal(source: str = "station", boot: int = 0, lines: int = 300, problems: bool = False) -> dict:
    if source not in system.JOURNAL_SOURCES:
        raise HTTPException(status_code=422, detail=f"source must be one of {', '.join(system.JOURNAL_SOURCES)}")
    return system.journal(source, boot, max(1, min(lines, MAX_LINES)), problems)


@router.get("/api/v1/system/boots")
def boots() -> list[dict]:
    return system.boots()


@router.get("/api/v1/system/diagnostics.zip")
def diagnostics(request: Request) -> Response:
    """Everything worth sending with a bug report, in one file."""
    station = _station(request)
    store = station.store
    parts = {
        "status.json": station.status,
        "system.json": lambda: {**system.info(station.data_dir), "power": station.power_abilities(),
                                "service": system.under_systemd()},
        "camera.json": station.camera_state,
        "incidents.json": lambda: station.incidents,
        "next_set.json": lambda: station.next_set,
        "boots.json": system.boots,
        "recordings.json": lambda: [{k: v for k, v in s.items() if k not in ("path", "keyframes_us")}
                                    for s in station.recordings()],
        "sets.json": lambda: [brief(r) for r in store.list_results(1000)] if store is not None else [],
        "station-recent.txt": lambda: "\n".join(station.lines),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, get in parts.items():
            try:
                value = get()
                z.writestr(name, value if isinstance(value, str) else json.dumps(value, indent=1, default=str))
            except Exception as e:              # whatever fails, the rest still goes
                z.writestr(f"{name}.error.txt", repr(e))
        for path in logbook.files(station.log_dir):
            z.write(path, f"logs/{os.path.basename(path)}")
        for name, (source, boot, lines, problems) in JOURNALS.items():
            got = system.journal(source, boot, lines, problems)
            z.writestr(f"journal/{name}.txt", "\n".join(got["lines"]) if got["available"] else got["error"] or "")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"kvihtai-diagnostics-{socket.gethostname()}-{stamp}.zip"
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
