import io
import pytest
from PIL import Image
import numpy as np


def test_predict_valid_image(client, valid_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.png", valid_image_bytes, "image/png")},
    )
    assert response.status_code == 200


def test_predict_response_schema(client, valid_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.png", valid_image_bytes, "image/png")},
    )
    data = response.json()

    # Required top-level fields
    for field in ["prediction", "probability", "confidence",
                  "cdr", "cdr_risk", "processing_time_ms",
                  "images", "clinical_note"]:
        assert field in data, f"Missing field: {field}"

    # Types
    assert data["prediction"] in ("glaucoma", "normal")
    assert 0.0 <= data["probability"] <= 1.0
    assert data["confidence"] in ("high", "medium", "low")
    assert 0.0 <= data["cdr"] <= 1.0
    assert data["cdr_risk"] in ("elevated", "borderline", "normal")
    assert isinstance(data["processing_time_ms"], int)

    # Images present
    for img_field in ["original", "heatmap_overlay",
                      "disc_mask", "cup_mask", "segmentation_overlay"]:
        assert img_field in data["images"]
        assert len(data["images"][img_field]) > 0


def test_predict_invalid_extension(client):
    response = client.post(
        "/predict",
        files={"file": ("report.pdf", b"fake pdf content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()


def test_predict_too_large(client):
    # 11 MB of random bytes
    large_bytes = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("big.png", large_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "exceeds limit" in response.json()["detail"].lower()


def test_predict_corrupted_image(client):
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 400
    assert "could not be read" in response.json()["detail"].lower()


def test_predict_tiny_image(client):
    # 50×50 image — below 100×100 minimum
    img = Image.fromarray(
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("tiny.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 400
    assert "too small" in response.json()["detail"].lower()


def test_metrics_endpoint(client):
    response = client.get("/results/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 6
    assert "ablation" in data
    assert "segmentation" in data