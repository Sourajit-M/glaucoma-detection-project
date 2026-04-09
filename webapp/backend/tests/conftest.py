import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import io
from PIL import Image

# Node name used by Grad-CAM to request the intermediate layer4 feature maps
_LAYER4_NODE = "/backbone/layer4/layer4.1/relu_1/Relu_output_0"


def make_fake_session(output_shape: tuple, output_value: float = 0.0):
    """
    Creates a mock ONNX InferenceSession that returns a
    numpy array of the given shape filled with output_value.
    """
    mock = MagicMock()
    mock.run.return_value = [np.full(output_shape, output_value, dtype=np.float32)]
    return mock


def make_test_image_bytes(width: int = 200, height: int = 200) -> bytes:
    """Creates a minimal valid PNG image in memory."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_registry():
    """
    Patches the model registry with fake ONNX sessions.
    CNN returns logits [0.2, 2.0] → glaucoma probability ~0.85
    U-Nets return all-zero masks → CDR fallback 0.5
    """
    with patch("app.routers.predict.registry") as mock_reg:
        mock_reg.is_ready = True
        mock_reg.loaded_models = ["cnn", "disc_unet", "cup_unet"]

        # CNN: smart mock that handles both inference and Grad-CAM calls.
        # - inference: run(None, ...) or run(["logits"], ...) → [(1,2) logits]
        # - Grad-CAM:  run([LAYER4_NODE, "logits"], ...) → [(1,512,7,7) acts, (1,2) logits]
        logits_output = np.array([[0.2, 2.0]], dtype=np.float32)            # (1, 2)
        layer4_output = np.zeros((1, 512, 7, 7), dtype=np.float32)          # (1, 512, 7, 7)

        def cnn_run_side_effect(output_names, feed_dict):
            if isinstance(output_names, list) and _LAYER4_NODE in output_names:
                return [layer4_output, logits_output]
            return [logits_output]

        mock_reg.cnn = MagicMock()
        mock_reg.cnn.run.side_effect = cnn_run_side_effect

        # U-Nets: (1, 1, 256, 256) mask logits — all zeros (sigmoid → ~0) → no masks
        mock_reg.disc = make_fake_session((1, 1, 256, 256), -5.0)
        mock_reg.cup  = make_fake_session((1, 1, 256, 256), -5.0)

        yield mock_reg


@pytest.fixture
def client(mock_registry):
    """TestClient with mocked registry."""
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_image_bytes():
    return make_test_image_bytes()