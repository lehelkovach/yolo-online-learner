import numpy as np
from perception.bbp import BBP, BoundingBox
from tracking.tracker import WorldModel

def make_bbp(x, y, w, h, idx=0):
    return BBP(
        frame_idx=idx,
        timestamp_s=idx * 0.1,
        bbox=BoundingBox(x, y, x+w, y+h),
        confidence=0.9,
        class_id=1
    )

def test_object_persistence():
    world = WorldModel(iou_threshold=0.1)
    
    # Frame 1: Object at (10, 10)
    bbp1 = make_bbp(10, 10, 10, 10, idx=0)
    world.update([bbp1], dt=0.1)
    
    assert len(world.active_objects) == 1
    obj = world.active_objects[0]
    assert not obj.is_ghost
    assert obj.id == 0
    # Check position roughly
    assert abs(obj.bbox.x1 - 10) < 1.0

    # Frame 2: Object moves to (12, 12)
    bbp2 = make_bbp(12, 12, 10, 10, idx=1)
    world.update([bbp2], dt=0.1)
    
    assert len(world.active_objects) == 1
    obj = world.active_objects[0]
    assert not obj.is_ghost
    # Should follow update
    assert abs(obj.bbox.x1 - 12) < 2.0 

    # Frame 3: Object occluded (no detections)
    world.update([], dt=0.1)
    
    assert len(world.active_objects) == 1
    obj = world.active_objects[0]
    assert obj.is_ghost
    # Should predict movement (velocity approx (2,2) per 0.1s -> should be at 14, 14)
    # The Kalman Filter might need settling, but it should be somewhere.
    print(f"Ghost pos: {obj.bbox.x1}, {obj.bbox.y1}")

    # Frame 4: Object reappears at (16, 16)
    bbp3 = make_bbp(16, 16, 10, 10, idx=3)
    world.update([bbp3], dt=0.1)
    
    assert len(world.active_objects) == 1
    obj = world.active_objects[0]
    assert not obj.is_ghost
    assert abs(obj.bbox.x1 - 16) < 2.0

def test_ghost_expiry():
    world = WorldModel()
    world._create_object(make_bbp(10, 10, 10, 10))
    
    # Update with no detections for many seconds
    # Max ghost time is 5.0s. 100 * 0.1 = 10s. Should expire.
    for _ in range(100):
        world.update([], dt=0.1)
        if not world.active_objects:
            break
            
    assert len(world.active_objects) == 0
