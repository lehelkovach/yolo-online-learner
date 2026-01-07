from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class YoloDetection:
    bbox: BoundingBox
    confidence: float
    class_id: int


class YoloBbpGenerator:
    """
    Ultralytics YOLO adapter.

    This keeps YOLO weights frozen and converts detections into BBPs.
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        *,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: Sequence[int] | None = None,
        max_det: int = 300,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Install with: pip install ultralytics"
            ) from e

        self._YOLO = YOLO
        self.model = YOLO(model)
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = None if classes is None else list(map(int, classes))
        self.max_det = int(max_det)

    def detect(self, frame_bgr: object) -> list[YoloDetection]:
        """
        Run YOLO inference on a single frame (OpenCV BGR image).

        Returns a list of detections with absolute pixel boxes.
        """
        results = self.model.predict(
            frame_bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()

        dets: list[YoloDetection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
            dets.append(
                YoloDetection(
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(c),
                    class_id=int(k),
                )
            )
        return dets

    def to_bbps(
        self, *, frame_idx: int, timestamp_s: float, dets: Sequence[YoloDetection]
    ) -> list[BBP]:
        return [
            BBP(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                embedding=None,
                salience=None,
                novelty=None,
                prediction_error=None,
            )
            for d in dets
        ]

    def detect_bbps(
        self, *, frame_idx: int, timestamp_s: float, frame_bgr: object
    ) -> list[BBP]:
        dets = self.detect(frame_bgr)
        return self.to_bbps(frame_idx=frame_idx, timestamp_s=timestamp_s, dets=dets)


def bbp_summary(bbps: Sequence[BBP]) -> tuple[int, float, int | None]:
    """
    Small helper for logging/telemetry.

    Returns: (count, max_confidence, most_common_class_id)
    """
    if not bbps:
        return (0, 0.0, None)
    max_conf = max(b.confidence for b in bbps)
    counts: dict[int, int] = {}
    for b in bbps:
        if b.class_id is None:
            continue
        counts[b.class_id] = counts.get(b.class_id, 0) + 1
    most_common = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    return (len(bbps), float(max_conf), most_common)

