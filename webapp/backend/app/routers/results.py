import json
from pathlib import Path
from fastapi import APIRouter
from app.schemas.prediction import ResultsResponse

router = APIRouter()

# Load once at import time — data never changes
_METRICS_PATH = Path(__file__).parent.parent.parent / "data" / "metrics.json"
_metrics_cache: dict | None = None


def _load_metrics() -> dict:
    global _metrics_cache
    if _metrics_cache is None:
        with open(_METRICS_PATH) as f:
            _metrics_cache = json.load(f)
    return _metrics_cache


@router.get("/results/metrics")
async def get_metrics():
    return _load_metrics()