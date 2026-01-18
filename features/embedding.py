from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.bbp import BoundingBox


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    include_geometry: bool = True
    include_intensity: bool = True


class EmbeddingExtractor:
    """
    Minimal embedding extractor from fovea + optional geometry.
    """

    def __init__(self, *, config: EmbeddingConfig | None = None) -> None:
        self.config = config if config is not None else EmbeddingConfig()

    def extract(
        self,
        fovea: np.ndarray,
        *,
        bbox: BoundingBox | None,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        features: list[float] = []
        if self.config.include_intensity:
            feats = _intensity_stats(fovea)
            features.extend(feats)

        if self.config.include_geometry and bbox is not None:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            cx = (bbox.x1 + bbox.x2) / 2.0 / max(1, width)
            cy = (bbox.y1 + bbox.y2) / 2.0 / max(1, height)
            bw = (bbox.x2 - bbox.x1) / max(1, width)
            bh = (bbox.y2 - bbox.y1) / max(1, height)
            features.extend([float(cx), float(cy), float(bw), float(bh)])

        return np.asarray(features, dtype=np.float32)

    def output_dim(self, *, num_channels: int, include_geometry: bool | None = None) -> int:
        include_geometry = self.config.include_geometry if include_geometry is None else include_geometry
        include_intensity = self.config.include_intensity
        dim = 0
        if include_intensity:
            dim += max(1, int(num_channels)) * 2
        if include_geometry:
            dim += 4
        return dim


def _intensity_stats(fovea: np.ndarray) -> list[float]:
    if fovea.ndim == 2:
        mean = float(np.mean(fovea))
        std = float(np.std(fovea))
        return [mean, std]
    channels = fovea.shape[2]
    feats: list[float] = []
    for i in range(channels):
        channel = fovea[:, :, i]
        feats.append(float(np.mean(channel)))
        feats.append(float(np.std(channel)))
    return feats
