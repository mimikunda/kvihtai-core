from datetime import datetime, timezone
from fastapi import APIRouter
from app.api.schemas import SetSummary, Rep, RepMetrics, Coordinate

router = APIRouter(tags=["sets"])

@router.get("/api/v1/sets/latest", response_model=SetSummary)
def get_latest_set() -> SetSummary:
    return SetSummary(
        set_id="mock-uuid-v4",
        timestamp=datetime.now(timezone.utc),
        processed_video_url="http://192.168.4.1:8080/static/renders/set_mock.mp4",
        reps=[
            Rep(
                rep_number=1,
                metrics=RepMetrics(
                    mean_concentric_velocity_mps=1.42,
                    peak_velocity_mps=2.10,
                    horizontal_loop_deviation_mm=38.5,
                    vertical_displacement_mm=1120.0
                ),
                trajectory=[
                    Coordinate(x_mm=0.0, y_mm=225.0, t_ms=0.0, vy_mps=0.0),
                    Coordinate(x_mm=2.1, y_mm=340.2, t_ms=33.3, vy_mps=1.2),
                    Coordinate(x_mm=-15.4, y_mm=1100.5, t_ms=1120.0, vy_mps=0.0)
                ]
            )
        ]
    )