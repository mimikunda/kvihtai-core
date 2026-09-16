"""Application settings.

Server binding lives in `run.sh`, not here. This holds only what the
application itself needs. Kept dependency-free on purpose: the daemon runs on
constrained edge hardware.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "KvihtAI Core"
    version: str = "0.0.0"


settings = Settings()
