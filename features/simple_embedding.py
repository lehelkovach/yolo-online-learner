from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perception.bbp import BoundingBox

SIMPLE_EMBEDDING_SPACE_ID = "simple_crop_v1"
SIMPLE_EMBEDDING_DIM = 10
SIMPLE_EMBEDDING_FEATURES = (
    "bbox_width_over_frame_width",
    "bbox_height_over_frame_height",
    "bbox_area_over_frame_area",
    "bbox_width_over_width_plus_height",
    "crop_mean_r_over_255",
    "crop_mean_g_over_255",
    "crop_mean_b_over_255",
    "crop_std_r_over_127_5",
    "crop_std_g_over_127_5",
    "crop_std_b_over_127_5",
)

EmbeddingMetrics = dict[str, object]


@dataclass(frozen=True, slots=True)
class SimpleCropEmbedding:
    """A deterministic embedding and the integer crop used to produce it."""

    vector: tuple[float, ...]
    crop_xyxy: tuple[int, int, int, int]
    raw_norm: float


def simple_embedding_schema() -> dict[str, object]:
    """Return the immutable schema recorded in every Stage-3 session."""
    return {
        "embedding_space_id": SIMPLE_EMBEDDING_SPACE_ID,
        "dimension": SIMPLE_EMBEDDING_DIM,
        "feature_order": list(SIMPLE_EMBEDDING_FEATURES),
        "normalization": "l2",
        "input_channel_order": "bgr",
        "statistics_channel_order": "rgb",
        "crop_rasterization": "floor_start_ceil_stop",
        "scope": "wta_winner_only",
    }


def attended_embedding_metrics(
    result: SimpleCropEmbedding | None,
    *,
    selected_bbp_index: int | None,
) -> EmbeddingMetrics:
    """Return a fixed-key per-frame metric payload for Stage 3."""
    status = "ok" if result is not None else (
        "no_selection" if selected_bbp_index is None else "invalid_crop"
    )
    return {
        "embedding_space_id": SIMPLE_EMBEDDING_SPACE_ID,
        "status": status,
        "selected_bbp_index": selected_bbp_index,
        "dimension": SIMPLE_EMBEDDING_DIM,
        "norm": None if result is None else float(np.linalg.norm(result.vector)),
        "raw_norm": None if result is None else result.raw_norm,
        "crop_xyxy": None if result is None else list(result.crop_xyxy),
    }


def embed_attended_crop(
    frame_bgr: object,
    bbox: BoundingBox,
) -> SimpleCropEmbedding | None:
    """Encode one attended crop with bounded geometry and simple RGB statistics.

    Geometry uses the continuously clipped box. Pixel statistics use a crop whose
    start coordinates are floored and stop coordinates are ceiled.
    """
    if not isinstance(frame_bgr, np.ndarray):
        return None
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return None

    frame_height, frame_width = frame_bgr.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None

    coordinates = np.asarray(bbox.as_xyxy(), dtype=np.float64)
    if not np.isfinite(coordinates).all():
        return None
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        return None

    x1 = max(0.0, min(float(frame_width), bbox.x1))
    y1 = max(0.0, min(float(frame_height), bbox.y1))
    x2 = max(0.0, min(float(frame_width), bbox.x2))
    y2 = max(0.0, min(float(frame_height), bbox.y2))
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None

    crop_x1 = max(0, min(frame_width, math.floor(x1)))
    crop_y1 = max(0, min(frame_height, math.floor(y1)))
    crop_x2 = max(0, min(frame_width, math.ceil(x2)))
    crop_y2 = max(0, min(frame_height, math.ceil(y2)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    pixel_count = crop.shape[0] * crop.shape[1]
    sums_bgr = np.sum(crop, axis=(0, 1), dtype=np.float64)
    squared_sums_bgr = np.einsum(
        "hwc,hwc->c",
        crop,
        crop,
        dtype=np.float64,
    )
    means_bgr = sums_bgr / pixel_count
    variances_bgr = np.maximum((squared_sums_bgr / pixel_count) - means_bgr**2, 0.0)
    stds_bgr = np.sqrt(variances_bgr)
    means_rgb = means_bgr[::-1] / 255.0
    stds_rgb = stds_bgr[::-1] / 127.5

    raw = np.asarray(
        [
            width / frame_width,
            height / frame_height,
            (width * height) / (frame_width * frame_height),
            width / (width + height),
            *means_rgb,
            *stds_rgb,
        ],
        dtype=np.float64,
    )
    if raw.shape != (SIMPLE_EMBEDDING_DIM,) or not np.isfinite(raw).all():
        return None

    raw_norm = float(np.linalg.norm(raw))
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        return None
    normalized = raw / raw_norm
    return SimpleCropEmbedding(
        vector=tuple(float(value) for value in normalized),
        crop_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
        raw_norm=raw_norm,
    )
