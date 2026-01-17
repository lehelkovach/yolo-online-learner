from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.bbp import BoundingBox


@dataclass(frozen=True, slots=True)
class FoveaSample:
    center: tuple[float, float]
    bbox: BoundingBox
    fovea: np.ndarray
    periphery: np.ndarray


class RetinaSampler:
    """
    Simple retina sampler with a high-resolution fovea and low-resolution periphery.
    """

    def __init__(
        self,
        *,
        fovea_size: tuple[int, int] = (96, 96),
        periphery_stride: int = 8,
        gaussian_sigma: float = 24.0,
    ) -> None:
        self.fovea_size = (int(fovea_size[0]), int(fovea_size[1]))
        self.periphery_stride = max(1, int(periphery_stride))
        self.gaussian_sigma = float(gaussian_sigma)

    def sample(self, frame: np.ndarray, *, center: tuple[float, float]) -> FoveaSample:
        height, width = frame.shape[:2]
        fovea_w, fovea_h = self.fovea_size
        cx = float(_clamp(center[0], 0.0, width - 1))
        cy = float(_clamp(center[1], 0.0, height - 1))

        x1 = int(round(cx - fovea_w / 2))
        y1 = int(round(cy - fovea_h / 2))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = min(width, x1 + fovea_w)
        y2 = min(height, y1 + fovea_h)
        x1 = max(0, x2 - fovea_w)
        y1 = max(0, y2 - fovea_h)

        fovea = _crop_with_padding(frame, x1, y1, fovea_w, fovea_h)
        periphery = frame[:: self.periphery_stride, :: self.periphery_stride].copy()

        bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
        return FoveaSample(center=(cx, cy), bbox=bbox, fovea=fovea, periphery=periphery)


def _crop_with_padding(
    frame: np.ndarray, x1: int, y1: int, width: int, height: int
) -> np.ndarray:
    crop = frame[y1 : y1 + height, x1 : x1 + width]
    if crop.shape[0] == height and crop.shape[1] == width:
        return crop.copy()

    channels = 1 if frame.ndim == 2 else frame.shape[2]
    pad_shape = (height, width) if channels == 1 else (height, width, channels)
    padded = np.zeros(pad_shape, dtype=frame.dtype)
    padded[: crop.shape[0], : crop.shape[1]] = crop
    return padded


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))
