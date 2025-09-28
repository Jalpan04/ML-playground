# train_model.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os


# --- MODIFIED FUNCTION ---
def train_and_save_model(n_clusters: int = 4):  # Set default to optimal k
    """
    Loads pattern data, trains a K-Means model, and saves it.
    """
    try:
        df = pd.read_csv("patterns.csv")
    except FileNotFoundError:
        print("Error: patterns.csv not found. Please run data_generation.py first.")
        return

    df.dropna(inplace=True)

    if df.empty:
        print("Error: patterns.csv is empty. Data generation might have failed.")
        return

    features = ["n_live_cells", "bounding_box_area", "density", "periodicity", "center_of_mass_change"]
    X = df[features].values  # Use .values to avoid feature name warnings

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    df['cluster'] = kmeans.labels_
    print("Cluster Analysis (Mean Feature Values):")
    cluster_analysis = df.groupby('cluster')[features].mean()
    print(cluster_analysis)

    # Manual mapping based on the analysis in Analysis.ipynb
    # This logic should be adapted based on your specific cluster results.
    cluster_labels = {}
    for i in range(n_clusters):
        cluster_data = cluster_analysis.loc[i]
        label = "Unknown"
        if cluster_data['periodicity'] > 1.5 or cluster_data['center_of_mass_change'] > 0.1:
            label = "Oscillator"
        else:  # It's a still life, let's categorize by size
            if cluster_data['n_live_cells'] > 12:
                label = "Large Still Life"
            elif cluster_data['n_live_cells'] > 6:
                label = "Medium Still Life"
            else:
                label = "Small Still Life"
        cluster_labels[i] = label

    print("\nAssigned Cluster Labels:")
    print(cluster_labels)

    output_dir = "saved_model"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(output_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(cluster_labels, os.path.join(output_dir, "cluster_labels.joblib"))

    print(f"\nModel, scaler, and labels saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    train_and_save_model()