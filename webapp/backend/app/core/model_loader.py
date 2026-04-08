import onnxruntime as ort
from pathlib import Path
from app.core.config import settings


class ModelRegistry:
    """
    Holds all three ONNX inference sessions.
    Loaded once at startup, reused for every request.
    """

    def __init__(self):
        self.cnn:  ort.InferenceSession | None = None
        self.disc: ort.InferenceSession | None = None
        self.cup:  ort.InferenceSession | None = None
        self._ready = False

    def load_all(self, base_dir: Path):
        """
        Load all models from base_dir.
        Called once inside FastAPI's lifespan startup event.
        """
        self.cnn  = self._load(base_dir / settings.cnn_model_path,  "CNN")
        self.disc = self._load(base_dir / settings.disc_model_path, "Disc U-Net")
        self.cup  = self._load(base_dir / settings.cup_model_path,  "Cup U-Net")
        self._ready = True

    def _load(self, path: Path, name: str) -> ort.InferenceSession:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found at {path}. "
                "Run export_onnx.py and copy the .onnx files to backend/models/"
            )
        print(f"  Loading {name} from {path.name} ...", flush=True)
        sess = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        print(f"  {name} loaded OK", flush=True)
        return sess

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def loaded_models(self) -> list[str]:
        names = []
        if self.cnn:  names.append("cnn")
        if self.disc: names.append("disc_unet")
        if self.cup:  names.append("cup_unet")
        return names


# Module-level singleton — import `registry` wherever you need a model
registry = ModelRegistry()