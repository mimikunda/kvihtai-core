"""Analysed sets: the newest in the contract's shape, and every set in full."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.api.schemas import SetSummary
from app.db import repository

router = APIRouter(tags=["sets"])


def _brief(result: dict) -> dict:
    """A set without its trajectory, for lists."""
    rises = [r for mv in result.get("movements", []) for r in mv["rises"]]
    return {
        "set_id": result["set_id"],
        "started_at": result["started_at"],
        "duration_ms": result.get("duration_ms"),
        "frames": result.get("frames"),
        "measured": result.get("measured"),
        "camera": result.get("camera"),
        "rises": rises,
        "has_video": bool(result.get("video")),
    }


@router.get("/api/v1/sets/latest", response_model=SetSummary)
def get_latest_set() -> SetSummary:
    set_summary = repository.get_latest_set()
    if set_summary is None:
        raise HTTPException(status_code=404, detail="No sets found")
    return set_summary


@router.get("/api/v1/sets")
def list_sets(limit: int = 100) -> list[dict]:
    return [_brief(r) for r in repository.list_results(limit)]


@router.get("/api/v1/sets/{set_id}")
def get_set(set_id: str) -> dict:
    result = repository.get_result(set_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No such set")
    result.pop("path", None)
    return result


@router.delete("/api/v1/sets/{set_id}")
def delete_set(set_id: str) -> dict:
    if not repository.delete_result(set_id):
        raise HTTPException(status_code=404, detail="No such set")
    return {"deleted": set_id}


@router.get("/api/v1/sets/{set_id}/clip.mp4")
def get_clip(set_id: str, request: Request):
    station = getattr(request.app.state, "station", None)
    path = station.clip_path(set_id) if station is not None else None
    if path is None:
        raise HTTPException(status_code=404, detail="No clip for this set")
    return FileResponse(path, media_type="video/mp4")
