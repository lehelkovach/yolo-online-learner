import numpy as np

from features.wta_layer import WTALayer


def test_wta_selects_winner_and_normalizes() -> None:
    layer = WTALayer(input_dim=4, num_units=3, k_winners=1, seed=0, learning_rate=0.5, decay=0.0)
    x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = layer.step(x)
    assert len(result.winners) == 1
    norm = float(np.linalg.norm(layer.weights[result.winners[0]]))
    assert norm <= 1.0001
