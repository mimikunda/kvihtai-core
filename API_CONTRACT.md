# KvihtAI Telemetry API Contract (v0.1.0)

All coordinates are in millimeters (mm) originating from the bottom-left of the calibrated capture frame. All velocities are in meters per second (m/s).

## 1. Live Telemetry Stream
**Endpoint:** `ws://<pi-ip>:8080/ws/live`
**Protocol:** WebSocket
**Frequency:** Per frame (e.g., 60Hz)

### Payload Schema (Server -> Client)
```json
{
  "timestamp_ms": 1726425600000,
  "frame_id": 142,
  "lift_state": "active", 
  "bar_center": {
    "x_mm": 24.5,
    "y_mm": 845.2
  },
  "kinematics": {
    "current_velocity_mps": 1.15,
    "acceleration_mps2": 9.8
  },
  "phase": "second_pull" 
}

```

*(Note: `lift_state` enum: `idle`, `active`, `completed`, `failed`)*

---

## 2. Set Analysis

**Endpoint:** `GET /api/v1/sets/latest`
**Protocol:** REST (HTTP/1.1)
**Purpose:** Fetches the aggregated kinematic breakdown immediately after a set finishes.

### Response Schema (200 OK)

```json
{
  "set_id": "uuid-v4",
  "timestamp": "2026-09-15T18:00:00Z",
  "metrics": {
    "mean_concentric_velocity_mps": 1.42,
    "peak_velocity_mps": 2.10,
    "horizontal_loop_deviation_mm": 38.5,
    "vertical_displacement_mm": 1120.0,
    "duration_ms": 1850
  },
  "trajectory": [
    {"x_mm": 0.0, "y_mm": 225.0, "t_ms": 0},
    {"x_mm": 2.1, "y_mm": 340.2, "t_ms": 33},
    {"x_mm": -15.4, "y_mm": 1100.5, "t_ms": 1120}
  ]
}

```
