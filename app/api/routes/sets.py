"""Analysed sets: the newest in the contract's shape, and every set in full."""

import csv
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.api.schemas import SetSummary
from app.db import repository

router = APIRouter(tags=["sets"])


class SetTags(BaseModel):
    """What the lifter says a set was. A field left out stays as it is; null clears it."""
    lift: str | None = None
    weight_kg: float | None = None
    note: str | None = None


def brief(result: dict) -> dict:
    """A set without its trajectory, for lists."""
    rises = [r for mv in result.get("movements", []) for r in mv["rises"]]
    return {
        "set_id": result["set_id"],
        "started_at": result["started_at"],
        "duration_ms": result.get("duration_ms"),
        "frames": result.get("frames"),
        "measured": result.get("measured"),
        "lost_frames": result.get("lost_frames", 0),
        "camera": result.get("camera"),
        "rises": rises,
        "has_video": bool(result.get("video")),
        "tags": result.get("tags"),
    }


def _station(request: Request):
    station = getattr(request.app.state, "station", None)
    if station is None:
        raise HTTPException(status_code=503, detail="The station is not running")
    return station


@router.get("/api/v1/sets/latest", response_model=SetSummary)
def get_latest_set() -> SetSummary:
    set_summary = repository.get_latest_set()
    if set_summary is None:
        raise HTTPException(status_code=404, detail="No sets found")
    return set_summary


@router.get("/api/v1/sets")
def list_sets(limit: int = 100) -> list[dict]:
    return [brief(r) for r in repository.list_results(limit)]


@router.get("/api/v1/sets.csv")
def export_sets(limit: int = 10000) -> Response:
    """Every rep of every set, one row each, for a spreadsheet."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["set_id", "date", "time", "lift", "weight_kg", "rep", "height_mm",
                     "mean_velocity_mps", "peak_velocity_mps", "rep_s", "set_s", "note"])
    for result in reversed(repository.list_results(limit)):
        when = datetime.fromisoformat(result["started_at"]).astimezone()
        tags = result.get("tags") or {}
        rises = [r for mv in result.get("movements", []) for r in mv["rises"]]
        for i, r in enumerate(rises, 1):
            writer.writerow([result["set_id"], when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S"),
                             tags.get("lift") or "", tags.get("weight_kg") or "", i, r["height_mm"],
                             r["mean_concentric_velocity_mps"], r["peak_velocity_mps"],
                             round((r["end_ms"] - r["start_ms"]) / 1000, 2),
                             round((result.get("duration_ms") or 0) / 1000, 1), tags.get("note") or ""])
    name = f"kvihtai-sets-{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/api/v1/sets/{set_id}")
def get_set(set_id: str) -> dict:
    result = repository.get_result(set_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No such set")
    result.pop("path", None)
    return result


@router.patch("/api/v1/sets/{set_id}")
def tag_set(set_id: str, tags: SetTags, request: Request) -> dict:
    result = _station(request).tag_set(set_id, tags.model_dump(exclude_unset=True))
    if result is None:
        raise HTTPException(status_code=404, detail="No such set")
    result.pop("path", None)
    return result


@router.delete("/api/v1/sets/{set_id}")
def delete_set(set_id: str, request: Request) -> dict:
    if not _station(request).delete_set(set_id):
        raise HTTPException(status_code=404, detail="No such set")
    return {"deleted": set_id}


@router.get("/api/v1/sets/{set_id}/clip.mp4")
def get_clip(set_id: str, request: Request):
    station = getattr(request.app.state, "station", None)
    path = station.clip_path(set_id) if station is not None else None
    if path is None:
        raise HTTPException(status_code=404, detail="No clip for this set")
    return FileResponse(path, media_type="video/mp4")
