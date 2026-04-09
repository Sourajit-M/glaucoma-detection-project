from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model_loader import registry
from app.routers import health, predict, results


# ── Lifespan: runs on startup and shutdown ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — load all models before accepting requests
    print("Loading models...", flush=True)
    base_dir = Path(__file__).parent.parent   # webapp/backend/
    registry.load_all(base_dir)
    print(f"Models ready: {registry.loaded_models}", flush=True)

    yield   # server runs here, handling requests

    # SHUTDOWN — nothing to clean up for ONNX sessions
    print("Shutting down.", flush=True)


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GlaucomaDetect API",
    description=(
        "Glaucoma detection from retinal fundus images using "
        "ResNet18 + U-Net CDR fusion. Research tool — not for clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────
app.include_router(health.router,  tags=["health"])
app.include_router(predict.router, tags=["inference"])
app.include_router(results.router, tags=["results"])