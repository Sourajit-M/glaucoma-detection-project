from pydantic import BaseModel, Field
from typing import Literal


class ImageSet(BaseModel):
    """Base64-encoded PNG images returned with every prediction."""

    original: str = Field(
        description="Original uploaded image, resized to 224×224, base64 PNG"
    )
    heatmap_overlay: str = Field(
        description="Grad-CAM heatmap blended over the original, base64 PNG"
    )
    disc_mask: str = Field(
        description="Binary optic disc segmentation mask, base64 PNG"
    )
    cup_mask: str = Field(
        description="Binary optic cup segmentation mask, base64 PNG"
    )
    segmentation_overlay: str = Field(
        description="Original image with disc (green) and cup (red) painted on, base64 PNG"
    )


class PredictionResponse(BaseModel):
    """Full response from POST /predict."""

    prediction: Literal["glaucoma", "normal"] = Field(
        description="Binary classification result"
    )
    probability: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence that this image shows glaucoma (0–1)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high ≥ 0.80 · medium ≥ 0.60 · low < 0.60"
    )
    cdr: float = Field(
        ge=0.0, le=1.0,
        description="Cup-to-disc ratio computed from U-Net segmentation"
    )
    cdr_risk: Literal["elevated", "borderline", "normal"] = Field(
        description="elevated ≥ 0.65 · borderline ≥ 0.50 · normal < 0.50"
    )
    processing_time_ms: int = Field(
        description="Total server-side processing time in milliseconds"
    )
    images: ImageSet = Field(
        description="All visualisation images as base64 PNG strings"
    )
    clinical_note: str = Field(
        description="Disclaimer reminding the user this is a research tool"
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: Literal["healthy", "degraded"] = Field(
        description="healthy = all models loaded · degraded = one or more missing"
    )
    models_loaded: list[str] = Field(
        description="Names of successfully loaded models"
    )
    version: str = Field(default="1.0.0")


class MetricEntry(BaseModel):
    name: str
    type: Literal["classical_ml", "deep_learning", "hybrid"]
    auc: float
    sensitivity: float
    specificity: float
    f1: float


class SegmentationEntry(BaseModel):
    structure: str
    dice: float
    iou: float


class AblationEntry(BaseModel):
    variant: str
    auc: float


class ResultsResponse(BaseModel):
    """Response from GET /results/metrics."""

    dataset_info: dict
    models: list[MetricEntry]
    ablation: list[AblationEntry]
    segmentation: list[SegmentationEntry]
    roc_curves: dict


class ErrorResponse(BaseModel):
    """Shape of all 4xx and 5xx error responses."""

    detail: str = Field(description="Human-readable error message")