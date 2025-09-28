# feature_extraction.py
import numpy as np
from scipy.ndimage import label, find_objects, center_of_mass
from typing import List, Dict, Any, Tuple


# isolate_patterns and calculate_features functions remain unchanged...
def isolate_patterns(grid: np.ndarray) -> List[np.ndarray]:
    """Isolates distinct connected patterns from a grid."""
    labeled_array, num_features = label(grid)
    if num_features == 0:
        return []
    objects = find_objects(labeled_array)
    patterns = []
    for i in range(num_features):
        loc = objects[i]
        pattern = grid[loc]
        patterns.append(pattern)
    return patterns


def calculate_features(pattern_cycle: List[np.ndarray]) -> Dict[str, Any]:
    """Calculates features for a pattern cycle."""
    first_pattern = pattern_cycle[0]
    n_live_cells = np.sum(first_pattern)
    if n_live_cells == 0:
        return None
    bounding_box_area = first_pattern.shape[0] * first_pattern.shape[1]
    density = n_live_cells / bounding_box_area if bounding_box_area > 0 else 0
    period = len(pattern_cycle)
    if period > 1:
        com_points = [center_of_mass(p) for p in pattern_cycle if np.sum(p) > 0]
        if len(com_points) > 1:
            com_points = np.array(com_points)
            center_of_mass_change = np.std(com_points, axis=0).sum()
        else:
            center_of_mass_change = 0.0
    else:
        center_of_mass_change = 0.0
    return {
        "n_live_cells": n_live_cells,
        "bounding_box_area": bounding_box_area,
        "density": density,
        "periodicity": period,
        "center_of_mass_change": center_of_mass_change
    }


# --- MODIFIED FUNCTION ---
def get_features_from_cycle(grid_cycle: List[np.ndarray]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Extracts all patterns, their features, and their bounding box slices from a cycle of grids.
    """
    first_grid = grid_cycle[0]
    labeled_array, num_patterns = label(first_grid)

    if num_patterns == 0:
        return [], []

    object_slices = find_objects(labeled_array)
    all_features = []
    valid_slices = []

    for i in range(num_patterns):
        obj_slice = object_slices[i]
        pattern_cycle = [grid[obj_slice] for grid in grid_cycle]

        features = calculate_features(pattern_cycle)
        if features:
            all_features.append(features)
            valid_slices.append(obj_slice)

    return all_features, valid_slices