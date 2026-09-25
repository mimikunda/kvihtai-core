# KvihtAI Core

Edge computer vision engine and kinematic analysis daemon for real-time barbell tracking on embedded systems.

`kvihtai-core` processes high-framerate video feeds directly on edge hardware, performs barbell/plate tracking, computes kinematic metrics (displacement, velocity, acceleration), segments lifting phases, and exposes telemetry over a local API.

---

## Features

- **Edge Computer Vision:** Low-latency plate and bar tracking optimized for embedded accelerators and hardware video decoders.
- **Kinematics Engine:** Instant calculation of mean concentric velocity (MCV), peak velocity (PV), vertical displacement, horizontal loop deviation, and acceleration profiles.
- **Phase Detection:** Automatic segmentation of key movement phases (first pull, transition, second pull, turnover, catch/lockout).
- **Local API:** Built-in REST/WebSocket endpoints for streaming live trajectories and serving set summaries to local network clients.

---

## Architecture


```

[ Camera Stream ]
       │
       ▼
[ Preprocessing & Undistortion ]
       │
       ▼
[ Tracker Engine (Plate/Bar Detection) ]
       │
       ▼
[ Kinematics & Phase Segmentation ]
 │
 ├──► [ Local SQLite Storage ]
 └──► [ Local REST / WebSocket Server ] ──► (Clients)

```

---

> **Note:** This project is in active early-stage development (WIP). Tracking pipelines, model weights, and API schemas are subject to breaking changes. Setup guides, benchmarks, and installation instructions will be published as stable milestones are reached.

For the current Raspberry Pi setup and a list of what is and is not working,
see [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).
