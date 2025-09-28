# tests/test_game.py
import pytest
import numpy as np
from game_of_life import get_next_generation


def test_block_is_stable():
    """A 2x2 block is a still life and should not change."""
    block = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    next_gen = get_next_generation(block)
    assert np.array_equal(block, next_gen)


def test_blinker_oscillates():
    """A blinker should oscillate with a period of 2."""
    # Phase 1: Horizontal
    blinker_h = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    # Phase 2: Vertical
    blinker_v = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Test transition from horizontal to vertical
    next_gen_1 = get_next_generation(blinker_h)
    assert np.array_equal(blinker_v, next_gen_1)

    # Test transition back to horizontal
    next_gen_2 = get_next_generation(next_gen_1)
    assert np.array_equal(blinker_h, next_gen_2)


def test_underpopulation():
    """A single live cell should die."""
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    next_gen = get_next_generation(grid)
    assert np.sum(next_gen) == 0


def test_overpopulation():
    """A 3x3 block of live cells should have its center die."""
    grid = np.ones((3, 3), dtype=int)
    next_gen = get_next_generation(grid)
    # Center cell (1,1) should die due to 8 neighbors
    assert next_gen[1, 1] == 0