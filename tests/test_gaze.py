from vision.gaze import GazeController


def test_gaze_follows_target_no_jitter() -> None:
    gaze = GazeController(jitter_std=0.0, jitter_max=0.0, pull_strength=1.0, seed=0)
    state = gaze.step((10.0, 20.0), frame_shape=(100, 100))
    assert state.center == (10.0, 20.0)
    state2 = gaze.step((10.0, 20.0), frame_shape=(100, 100))
    assert state2.dwell_frames == 1


def test_gaze_resets_on_target_change() -> None:
    gaze = GazeController(jitter_std=0.0, jitter_max=0.0, pull_strength=1.0, seed=0)
    _ = gaze.step((10.0, 20.0), frame_shape=(100, 100))
    state = gaze.step((30.0, 40.0), frame_shape=(100, 100))
    assert state.dwell_frames == 0
