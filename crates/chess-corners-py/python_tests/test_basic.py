import numpy as np
import pytest

import chess_corners


def _checkerboard(square_size: int = 16, squares: int = 8) -> np.ndarray:
    grid = (np.indices((squares, squares)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(grid, np.ones((square_size, square_size), dtype=np.uint8)) * 255
    return board


def test_find_chess_corners_basic():
    img = _checkerboard(square_size=16, squares=8)
    cfg = chess_corners.ChessConfig()
    cfg.threshold_rel = 0.1
    cfg.min_cluster_size = 1

    corners = chess_corners.find_chess_corners(img, cfg)
    assert corners.dtype == np.float32
    assert corners.ndim == 2
    assert corners.shape[1] == 4
    assert corners.shape[0] > 0


def test_find_chess_corners_rejects_wrong_dtype():
    img = _checkerboard(square_size=16, squares=8).astype(np.float32)
    with pytest.raises(TypeError):
        chess_corners.find_chess_corners(img)
