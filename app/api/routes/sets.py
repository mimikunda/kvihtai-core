from fastapi import APIRouter, HTTPException

from app.api.schemas import SetSummary
from app.db import repository

router = APIRouter(tags=["sets"])


@router.get("/api/v1/sets/latest", response_model=SetSummary)
def get_latest_set() -> SetSummary:
    set_summary = repository.get_latest_set()
    if set_summary is None:
        raise HTTPException(status_code=404, detail="No sets found")
    return set_summary