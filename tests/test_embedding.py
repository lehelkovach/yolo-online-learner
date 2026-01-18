import numpy as np

from features.embedding import EmbeddingExtractor
from perception.bbp import BoundingBox


def test_embedding_includes_geometry_and_stats() -> None:
    fovea = np.ones((2, 2, 3), dtype=np.uint8) * 10
    bbox = BoundingBox(0, 0, 20, 20)
    extractor = EmbeddingExtractor()
    embedding = extractor.extract(fovea, bbox=bbox, frame_shape=(40, 40))
    assert embedding.shape[0] == 10
    assert float(embedding[0]) == 10.0
    assert float(embedding[1]) == 0.0
