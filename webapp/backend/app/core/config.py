from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Model paths
    cnn_model_path: str = "models/glaucoma_resnet18.onnx"
    disc_model_path: str = "models/disc_unet.onnx"
    cup_model_path: str = "models/cup_unet.onnx"

    # Image sizes
    image_size: int = 224
    seg_image_size: int = 256

    # Thresholds
    cdr_threshold: float = 0.65
    confidence_threshold: float = 0.5

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:5173"

    # Limits
    max_file_size_mb: int = 10
    allowed_extensions: str = "jpg,jpeg,png,bmp"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ── Derived properties ────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",")}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# Single instance — import this everywhere
settings = Settings()