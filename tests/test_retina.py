import numpy as np

from vision.retina import RetinaSampler


def test_retina_sample_shapes() -> None:
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    sampler = RetinaSampler(fovea_size=(20, 20), periphery_stride=10)
    sample = sampler.sample(frame, center=(40, 50))
    assert sample.fovea.shape == (20, 20, 3)
    assert sample.periphery.shape == (10, 8, 3)


def test_retina_padding_at_edges() -> None:
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    sampler = RetinaSampler(fovea_size=(20, 20), periphery_stride=5)
    sample = sampler.sample(frame, center=(0, 0))
    assert sample.fovea.shape == (20, 20, 3)
    assert sample.bbox.x1 >= 0
    assert sample.bbox.y1 >= 0
