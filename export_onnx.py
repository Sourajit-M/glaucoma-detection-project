"""
export_onnx.py
==============
Run this once from the project root to export all three
PyTorch models to ONNX format for backend serving.

Usage:
    uv run python export_onnx.py

Outputs (copy these to webapp/backend/models/):
    outputs/models/glaucoma_resnet18.onnx
    outputs/models/disc_unet.onnx
    outputs/models/cup_unet.onnx
"""

import torch
from pathlib import Path

from config import MODELS_DIR, DEVICE
from models.cnn_model import build_model          # your existing function
from models.unet import build_unet                # your existing function


def export_cnn(ckpt_path: Path, out_path: Path):
    print(f"Exporting CNN from {ckpt_path.name} ...")

    # 1. Build model architecture (same as training — pretrained=False because
    #    we are loading weights, not downloading ImageNet weights again)
    model = build_model(pretrained=False, freeze_backbone=False).to("cpu")

    # 2. Load the saved checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. Set to eval mode — disables dropout and batchnorm training behaviour
    model.eval()

    # 4. Dummy input — batch=1, RGB, 224×224
    dummy = torch.randn(1, 3, 224, 224)

    # 5. Export
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"  Saved -> {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


def export_unet(ckpt_path: Path, out_path: Path, name: str):
    print(f"Exporting {name} U-Net from {ckpt_path.name} ...")

    # Build U-Net — same architecture used during training
    model = build_unet(encoder="resnet18", pretrained=False).to("cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # U-Net takes 256×256 input
    dummy = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["mask_logits"],
        dynamic_axes={"image": {0: "batch_size"}, "mask_logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"  Saved -> {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


def verify_onnx(onnx_path: Path, input_shape: tuple):
    """Quick sanity check — load the ONNX model and run one inference."""
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(*input_shape).astype(np.float32)
    out = sess.run(None, {"image": dummy})
    print(f"  Verified {onnx_path.name} — output shape: {out[0].shape}  OK")


if __name__ == "__main__":
    # ── Paths ────────────────────────────────────────────────────────
    cnn_ckpt  = MODELS_DIR / "glaucoma_resnet18_best.pth"
    disc_ckpt = MODELS_DIR / "disc_unet_best.pth"
    cup_ckpt  = MODELS_DIR / "cup_unet_best.pth"

    cnn_out   = MODELS_DIR / "glaucoma_resnet18.onnx"
    disc_out  = MODELS_DIR / "disc_unet.onnx"
    cup_out   = MODELS_DIR / "cup_unet.onnx"

    # ── Export ───────────────────────────────────────────────────────
    export_cnn(cnn_ckpt,  cnn_out)
    export_unet(disc_ckpt, disc_out, "disc")
    export_unet(cup_ckpt,  cup_out,  "cup")

    # ── Verify ───────────────────────────────────────────────────────
    print("\nVerifying exported models...")
    verify_onnx(cnn_out,  (1, 3, 224, 224))
    verify_onnx(disc_out, (1, 3, 256, 256))
    verify_onnx(cup_out,  (1, 3, 256, 256))

    print("\nAll done. Copy these to webapp/backend/models/:")
    for p in [cnn_out, disc_out, cup_out]:
        print(f"  {p}")