# app_pix.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import os
import time

from game_of_life import get_next_generation
from feature_extraction import get_features_from_cycle
from patterns import CLASSIC_PATTERNS

# --- App Configuration ---
st.set_page_config(page_title="Cellular Automata Classifier", layout="wide")


# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    model_path = os.path.join("saved_model", "kmeans_model.joblib")
    scaler_path = os.path.join("saved_model", "scaler.joblib")
    labels_path = os.path.join("saved_model", "cluster_labels.joblib")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, labels_path]):
        st.error("Model files not found. Please run `train_model.py` first.")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    cluster_labels = joblib.load(labels_path)
    return model, scaler, cluster_labels


kmeans_model, scaler, cluster_labels = load_model()

# --- Initialize Session State ---
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
    st.session_state.grid_history = []
    st.session_state.all_classifications = []  # NEW: Store classifications for every frame
    st.session_state.status_message = ""


# --- Helper Functions ---
def initialize_grid(grid_size, pattern_name, density):
    if pattern_name == "Random":
        return np.random.choice([0, 1], size=(grid_size, grid_size), p=[1 - density, density])
    else:
        grid = np.zeros((grid_size, grid_size), dtype=int)
        pattern = CLASSIC_PATTERNS[pattern_name]
        p_h, p_w = pattern.shape
        start_row = (grid_size - p_h) // 2
        start_col = (grid_size - p_w) // 2
        grid[start_row:start_row + p_h, start_col:start_col + p_w] = pattern
        return grid


def plot_grid(ax, grid, title, classifications=None, show_highlights=False, show_labels=False):
    ax.clear()
    ax.imshow(grid, cmap='binary')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if classifications and (show_highlights or show_labels):
        color_map = {"Small Still Life": "cyan", "Medium Still Life": "green", "Large Still Life": "lime",
                     "Oscillator": "orange", "Unknown": "red"}
        classifications_for_plot = [(c[0], c[1]) for c in classifications]
        for label, box_slice in classifications_for_plot:
            color = color_map.get(label, "gray")
            ymin, xmin = box_slice[0].start, box_slice[1].start
            height = box_slice[0].stop - ymin
            width = box_slice[1].stop - xmin

            if show_highlights:
                rect = patches.Rectangle((xmin - 0.5, ymin - 0.5), width, height, linewidth=2, edgecolor=color,
                                         facecolor='none')
                ax.add_patch(rect)

            if show_labels:
                ax.text(xmin - 0.5, ymin - 1.5, label, color=color, fontsize=10, weight='bold')


def get_classification_results(grid_state):
    # This function now takes a single grid state, not a cycle
    classifications = []
    # We treat the single state as a cycle of period 1 for feature extraction
    features_list, box_slices = get_features_from_cycle([grid_state])
    for i, features in enumerate(features_list):
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        scaled_features = scaler.transform(feature_vector)
        cluster_id = kmeans_model.predict(scaled_features)[0]
        label = cluster_labels.get(cluster_id, "Unknown")
        classifications.append((label, box_slices[i], features))
    return classifications


def display_classification_details(classifications):
    if not classifications:
        st.info("No patterns identified in this generation.")
    else:
        st.subheader(f"Detailed Analysis ({len(classifications)} patterns found)")
        for i, (label, _, features) in enumerate(classifications):
            with st.expander(f"**Pattern {i + 1}:** Classified as **{label}**"):
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(label="Live Cells", value=features["n_live_cells"])
                m_col2.metric(label="Periodicity", value=f"{features['periodicity']} gens")
                m_col3.metric(label="Density", value=f"{features['density']:.2f}")


# --- UI Components ---
st.title("Evolving Cellular Automata Classifier")

tab1, tab2 = st.tabs(["Simulation Viewer", "Classification Results"])

with tab1:
    st.subheader("Live Simulation & Review")
    live_plot_placeholder = st.empty()
    review_plot_placeholder = st.empty()

with st.sidebar:
    st.header("Simulation Controls")
    pattern_options = ["Random"] + list(CLASSIC_PATTERNS.keys())
    selected_pattern = st.selectbox("Choose a starting pattern:", pattern_options)
    GRID_SIZE = st.slider("Grid Size", 30, 100, 50)
    MAX_GENERATIONS = st.slider("Max Generations", 50, 500, 200)
    INITIAL_DENSITY = st.slider("Initial Density (for Random)", 0.1, 0.9, 0.2, disabled=(selected_pattern != "Random"))

    if st.button("Run Simulation"):
        st.session_state.simulation_run = True
        st.session_state.grid_history = []
        st.session_state.all_classifications = []  # Reset classifications history
        initial_grid = initialize_grid(GRID_SIZE, selected_pattern, INITIAL_DENSITY)
        current_grid = initial_grid.copy()
        byte_history = []
        stabilized = False
        fig, ax = plt.subplots()

        # --- Live Animation & Computation Phase ---
        # NOTE: This will be slower as classification runs on every frame.
        for gen in range(MAX_GENERATIONS + 1):
            st.session_state.grid_history.append(current_grid.copy())

            # MODIFIED: Classify the current grid state on every generation
            current_classifications = get_classification_results(current_grid)
            st.session_state.all_classifications.append(current_classifications)

            # Plot the current grid to show live animation
            plot_grid(ax, current_grid, f"Generation: {gen}")
            with live_plot_placeholder.container():
                st.pyplot(fig)
            time.sleep(0.05)

            current_bytes = current_grid.tobytes()
            if gen > 0 and current_bytes in byte_history:
                stabilized = True
                st.session_state.status_message = f"Stabilized at generation {gen}."
                break
            byte_history.append(current_bytes)
            current_grid = get_next_generation(current_grid)
        plt.close(fig)
        if not stabilized:
            st.session_state.status_message = f"Reached max {MAX_GENERATIONS} generations without stabilizing."

        live_plot_placeholder.empty()
        st.rerun()

# --- Post-Simulation Display Logic ---
if st.session_state.simulation_run and st.session_state.grid_history:
    with tab1:
        total_gens = len(st.session_state.grid_history) - 1
        slider_col, toggle1_col, toggle2_col = st.columns([3, 1, 1])
        with slider_col:
            selected_gen = st.slider("Review Generations:", 0, total_gens, total_gens)
        with toggle1_col:
            st.write("")
            show_highlights = st.checkbox("Show Boxes", value=True)
        with toggle2_col:
            st.write("")
            show_labels = st.checkbox("Show Labels", value=True)

        st.info(st.session_state.status_message)

        grid_to_display = st.session_state.grid_history[selected_gen]
        # MODIFIED: Get classifications for the SPECIFIC generation from the slider
        classifications_to_display = st.session_state.all_classifications[selected_gen]

        fig_review, ax_review = plt.subplots()

        # MODIFIED: No special case for the last frame; works for all frames
        plot_grid(ax_review, grid_to_display, f"Generation: {selected_gen}",
                  classifications=classifications_to_display,
                  show_highlights=show_highlights,
                  show_labels=show_labels)

        review_plot_placeholder.pyplot(fig_review)
        plt.close(fig_review)

    with tab2:
        # MODIFIED: Also uses classifications for the selected generation
        display_classification_details(st.session_state.all_classifications[selected_gen])