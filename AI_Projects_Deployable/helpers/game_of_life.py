# game_of_life.py

import numpy as np
from scipy.signal import convolve2d


def get_next_generation(grid: np.ndarray) -> np.ndarray:
    """
    Computes the next state of the grid according to Conway's Game of Life rules.

    Args:
        grid: A 2D NumPy array representing the current state (1 for live, 0 for dead).

    Returns:
        A 2D NumPy array for the next generation.
    """
    # Kernel for summing neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Use convolution to count live neighbors for each cell
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')

    # Apply Game of Life rules:
    # 1. A living cell with 2 or 3 neighbors survives.
    rule1 = (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))

    # 2. A dead cell with exactly 3 neighbors becomes a live cell.
    rule2 = (grid == 0) & (neighbor_count == 3)

    # Combine the rules to get the next state
    return (rule1 | rule2).astype(int)


def run_simulation_to_stabilization(grid: np.ndarray, max_generations: int = 200):
    """
    Runs the simulation until it stabilizes (repeating cycle or no change) or hits max generations.

    Args:
        grid: The initial 2D grid state.
        max_generations: The maximum number of steps to simulate.

    Returns:
        A tuple containing:
        - The final list of grids in the detected cycle.
        - The period of the cycle (1 for still life).
    """
    history = []
    current_grid = grid.copy()

    for _ in range(max_generations):
        # Store a hashable representation of the grid
        history.append(current_grid.tobytes())

        current_grid = get_next_generation(current_grid)

        # Check for stabilization
        current_bytes = current_grid.tobytes()
        if current_bytes in history:
            first_occurrence_index = history.index(current_bytes)
            cycle = [np.frombuffer(h, dtype=int).reshape(grid.shape) for h in history[first_occurrence_index:]]
            period = len(cycle)
            return cycle, period

    # If max generations reached without stabilization, return the last state
    return [current_grid], 1