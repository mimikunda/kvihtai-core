# Contributing to KvihtAI Core

This repository houses the edge computer vision pipeline, kinematics math, and the local telemetry daemon. Code here runs on constrained edge hardware. Performance, memory efficiency, and deterministic execution are non-negotiable.

## 1. Architectural Boundaries
- **CV Pipeline:** Keep computer vision processing strictly isolated from the REST/WebSocket networking layers.
- **Dependency Minimization:** Do not introduce heavy ML frameworks (e.g., PyTorch, TensorFlow) without explicit approval. Prefer OpenCV, TFLite, or ONNX runtimes optimized for ARM64 edge devices.
- **Hardware Agnosticism:** While tested on Raspberry Pi, the core tracking logic must not hardcode Pi-specific hardware endpoints unless wrapped in an interface.

## 2. Assisted Synthesis & LLM Usage
This project utilizes AI for engineering exploration. If you generate code using an LLM:
1. **Audit Every Line:** You are strictly responsible for the execution cost of the code.
2. **No Monoliths:** Do not paste 500-line generated files. Break down the logic into testable functions (e.g., a pure function that calculates velocity from two coordinate vectors).
3. **Memory Management:** AI tools often generate memory-leaking loops in CV pipelines (e.g., failing to release frame buffers). Verify resource management manually.

## 3. Git & Branching Protocol
1. **Atomic Commits:** Commit small, logical units. Do not commit the entire pipeline in a single `git push`.
2. **Branching:** Create feature branches off `main` (e.g., `feat/hough-circle-tracker`, `fix/websocket-latency`).
3. **Pull Requests:** Open a PR for integration. The repository maintainer will review the code for execution efficiency and schema compliance before merging.

## 4. Development Setup
- Use a strict virtual environment (`venv` or `uv`).
- Update `requirements.txt` immediately if a new dependency is required.
- Test endpoints against the JSON schemas defined in `docs/API_CONTRACT.md`.
