from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Coordinate(BaseModel):
    x_mm: float
    y_mm: float
    t_ms: float
    vy_mps: Optional[float] = None

class RepMetrics(BaseModel):
    mean_concentric_velocity_mps: float
    peak_velocity_mps: float
    horizontal_loop_deviation_mm: float
    vertical_displacement_mm: float

class Rep(BaseModel):
    rep_number: int
    metrics: RepMetrics
    trajectory: List[Coordinate]

class SetSummary(BaseModel):
    set_id: str
    timestamp: datetime
    processed_video_url: Optional[str] = None
    reps: List[Rep]