"""
VoxGaze Perception: Gaze Estimator Factory & Model Registry

Provides a decoupled factory for loading validated gaze estimators:
- 'mobileone_s0': Pretrained MobileOne S0 (fast, lightweight, edge-optimized)
- 'resnet34': Pretrained ResNet-34 (larger backbone, higher capacity)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from backend.app.perception.gaze_estimator_interface import BaseGazeEstimator
from backend.app.perception.mobileone_estimator import MobileOneGazeEstimator
from backend.app.perception.resnet34_estimator import ResNet34GazeEstimator

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"

DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "mobileone_s0": MODELS_DIR / "mobileone_s0_gaze.onnx",
    "resnet34": MODELS_DIR / "resnet34_gaze.onnx",
}

ESTIMATOR_CLASSES: Dict[str, Type[BaseGazeEstimator]] = {
    "mobileone_s0": MobileOneGazeEstimator,
    "resnet34": ResNet34GazeEstimator,
}


def get_available_models() -> List[str]:
    """Returns list of registered model architecture names."""
    return list(ESTIMATOR_CLASSES.keys())


def create_gaze_estimator(
    model_name: str,
    model_path: Optional[Union[str, Path]] = None,
    session: Optional[Any] = None,
) -> BaseGazeEstimator:
    """
    Instantiates a validated gaze estimator conforming to BaseGazeEstimator.

    Args:
        model_name: Identifier ('mobileone_s0' or 'resnet34').
        model_path: Optional custom path to model weights. Defaults to models/{model_name}_gaze.onnx.
        session: Optional pre-created ONNX runtime session (for testing/mocking).

    Returns:
        Instance conforming to BaseGazeEstimator.
    """
    model_key = model_name.lower().strip()
    if model_key not in ESTIMATOR_CLASSES:
        available = ", ".join(ESTIMATOR_CLASSES.keys())
        raise ValueError(f"Unknown model architecture '{model_name}'. Available: {available}")

    cls = ESTIMATOR_CLASSES[model_key]

    if session is not None:
        return cls(session=session)

    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATHS[model_key]
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Model weights for '{model_name}' not found at: {resolved_path}. "
            f"Please run the acquisition script: python scripts/download_{model_key.split('_')[0]}.py"
        )

    return cls(model_path=resolved_path)
