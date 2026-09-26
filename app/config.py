"""Application settings and environment configuration.

Every setting can be overridden by an environment variable with the prefix
KVIHTAI_, for example KVIHTAI_PORT=8081, or by the same line in a .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "KvihtAI Core"
    version: str = "0.1.0"

    # Network Binding
    host: str = "0.0.0.0"
    port: int = 8080

    # Where frames come from: "none" runs the API alone, "pi" the camera
    # module, "video:<path>" a recorded clip played at its own pace, over and over.
    camera: str = "none"
    camera_size: str = "1536x864"
    # 60, not the 80 tools/capture.py uses: the Pi 4B cannot record 80 fps
    # and keep every frame for the tracker too, see app.capture.source.
    target_fps: int = 60

    # Sets, recordings, the database and camera settings live here.
    data_dir: Path = ROOT / "data"
    # Record everything the camera sees, see app.capture.recorder.
    record: bool = True
    record_segment_s: float = 300.0
    record_min_free_gb: float = 5.0

    # The built web app (kvihtai-web's dist/), served at /. Nothing is served if it is missing.
    web_dir: Path = ROOT.parent / "kvihtai-web" / "dist"

    model_config = SettingsConfigDict(env_prefix="KVIHTAI_", env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")


settings = Settings()
