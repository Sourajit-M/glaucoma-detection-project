import numpy as np
from dataclasses import dataclass
import onnxruntime as ort


@dataclass
class PredictionResult:
    prediction: str        # "glaucoma" or "normal"
    probability: float     # glaucoma probability in [0, 1]
    confidence: str        # "high", "medium", or "low"


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Subtracting max prevents overflow with large logit values.
    """
    e = np.exp(logits - logits.max())
    return e / e.sum()


def get_confidence(probability: float) -> str:
    if probability >= 0.80:
        return "high"
    elif probability >= 0.60:
        return "medium"
    return "low"


def run_cnn(
    session: ort.InferenceSession,
    image_array: np.ndarray,       # (1, 3, 224, 224) float32
) -> PredictionResult:
    """
    Run CNN inference and return a structured prediction result.

    Args:
        session:     Loaded ONNX InferenceSession from ModelRegistry
        image_array: Preprocessed image from preprocess_for_cnn()

    Returns:
        PredictionResult with prediction, probability, and confidence
    """
    # Run ONNX inference — returns list of output arrays
    outputs = session.run(None, {"image": image_array})

    # outputs[0] shape: (1, 2) — one row, two logits
    logits = outputs[0][0]              # shape (2,) — remove batch dim

    probs  = softmax(logits)            # shape (2,) — sum to 1.0
    glaucoma_prob = float(probs[1])     # index 1 = glaucoma class

    prediction = "glaucoma" if glaucoma_prob >= 0.5 else "normal"
    confidence = get_confidence(glaucoma_prob)

    return PredictionResult(
        prediction=prediction,
        probability=round(glaucoma_prob, 4),
        confidence=confidence,
    )