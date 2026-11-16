import cv2
import numpy as np
import torch

from src.config import IMG_SIZE

VALID_VALUES = {0, 1, 2}
MIN_SIDE = 10


def validate_wafer(wafer_map) -> np.ndarray:
    """Check the input is a usable wafer map and return it as uint8."""
    m = np.asarray(wafer_map)

    if m.ndim != 2:
        raise ValueError(f"Wafer map must be 2D, got shape {m.shape}")
    if min(m.shape) < MIN_SIDE:
        raise ValueError(f"Wafer map must be at least {MIN_SIDE}x{MIN_SIDE}, got {m.shape}")

    found = set(np.unique(m).tolist())
    if not found <= VALID_VALUES:
        bad = sorted(found - VALID_VALUES)
        raise ValueError(f"Wafer map may only contain 0, 1, 2. Found: {bad}")

    return m.astype(np.uint8)


def resize_wafer(wafer_map, size: int = IMG_SIZE) -> np.ndarray:
    """Resize to size x size with nearest-neighbour, keeping values 0/1/2."""
    m = np.asarray(wafer_map, dtype=np.uint8)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def to_tensor(wafer: np.ndarray) -> torch.Tensor:
    """Scale 0/1/2 -> 0.0/0.5/1.0 and add the channel dimension: (1, H, W)."""
    img = wafer.astype(np.float32) / 2.0
    return torch.from_numpy(img).unsqueeze(0)


def preprocess(wafer_map, size: int = IMG_SIZE) -> torch.Tensor:
    """Full pipeline: validate -> resize -> scale -> tensor."""
    return to_tensor(resize_wafer(validate_wafer(wafer_map), size))