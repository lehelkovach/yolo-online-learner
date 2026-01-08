from perception.bbp import BBP, BoundingBox


def test_bbox_iou_identity() -> None:
    b = BoundingBox(0, 0, 10, 10)
    assert b.iou(b) == 1.0


def test_bbox_iou_disjoint() -> None:
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert a.iou(b) == 0.0


def test_bbp_roundtrip_dict() -> None:
    bbp = BBP(
        frame_idx=12,
        timestamp_s=0.4,
        bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
        confidence=0.9,
        class_id=7,
        embedding=(0.1, 0.2, 0.3),
        salience=0.5,
        novelty=0.6,
        prediction_error=0.7,
    )
    d = bbp.to_dict()
    bbp2 = BBP.from_dict(d)
    assert bbp2 == bbp

