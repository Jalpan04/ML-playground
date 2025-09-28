import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing  # Import the multiprocessing library

from game_of_life import run_simulation_to_stabilization
from feature_extraction import get_features_from_cycle


# --- Worker Function ---
# This function contains the logic for a single simulation run.
# It will be executed by each parallel process.
def run_single_simulation(args):
    """
    Runs one simulation and returns the features of resulting patterns.
    `args` is a tuple containing grid_size and initial_density.
    """
    grid_size, initial_density = args

    # Start with a random grid
    initial_grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1 - initial_density, initial_density])

    # Run until stable
    final_cycle, period = run_simulation_to_stabilization(initial_grid)

    # Extract features and return them
    features, _ = get_features_from_cycle(final_cycle)
    return features


# --- Main Data Generation Function ---
def generate_dataset(num_simulations: int, grid_size: int = 100):
    """
    Generates a dataset of features from multiple Game of Life simulations in parallel.
    """
    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Starting data generation with {num_simulations} simulations on {num_cores} CPU cores.")

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Prepare arguments for each simulation run
        simulation_args = [(grid_size, 0.2) for _ in range(num_simulations)]

        # Use pool.imap_unordered for efficient parallel processing with a progress bar
        # It distributes the `run_single_simulation` function across all processes
        results = []
        for result in tqdm(pool.imap_unordered(run_single_simulation, simulation_args), total=num_simulations,
                           desc="Running Simulations"):
            results.append(result)

    # Flatten the list of lists into a single list of features
    all_patterns_features = [item for sublist in results for item in sublist]

    # Create DataFrame and save
    df = pd.DataFrame(all_patterns_features)
    df.dropna(inplace=True)  # Clean any residual bad data
    df.to_csv("patterns.csv", index=False)
    print(f"\nSuccessfully generated patterns.csv with {len(df)} patterns.")


# The `if __name__ == "__main__":` guard is CRITICAL for multiprocessing to work correctly.
if __name__ == "__main__":
    generate_dataset(num_simulations=500, grid_size=30)