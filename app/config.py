"""Application settings and environment configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "KvihtAI Core"
    version: str = "0.1.0"
    
    # Network Binding
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Hardware limits & config
    camera_device: str = "/dev/video0"
    target_fps: int = 60
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()