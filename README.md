# ML Playground & Interactive Demos

An interactive Streamlit launcher hub that manages a suite of machine learning and computer vision web applications. Through a centralized dashboard, users can spin up individual microservices, monitor active ports, and explore various interactive algorithms.

## Featured Applications

### 🎨 Pixel Palette Extractor
- **Tech Stack**: Gradio, Pillow, NumPy
- **Description**: An image analysis tool designed to extract dominant color palettes from pixel art, sprites, and illustrations. Perfect for color quantization and artwork inspection.
- **Port**: `7860`

### ❌ Tic-Tac-Toe AI
- **Tech Stack**: Streamlit, scikit-learn, joblib
- **Description**: Visualizes search trees, Minimax game configurations, and predictions. Play against a neural-network or tree-search agent and watch decision-making statistics update in real-time.
- **Port**: `8502`

### 🧬 Cellular Automata Classifier
- **Tech Stack**: Streamlit, NumPy, Matplotlib
- **Description**: Simulates 2D cellular automata (Conway's Game of Life rulesets) and uses heuristic classification algorithms to identify emerging patterns (blinkers, gliders, still-lifes, noise).
- **Port**: `8503`

---

## The Launcher Hub (`hub.py`)

The root directory contains a Streamlit-based control center (`hub.py`) that acts as a process manager. 

- **Subprocess Management**: Spawn individual app servers in the background securely using Python `subprocess`.
- **Active Port Monitor**: Scans local network connections using `psutil` to verify if the server is listening before opening pages.
- **Graceful Shutdown**: Terminate parent and child process swarms when stopping servers.

## File Structure

```
├── hub.py                            # Centralized Streamlit server manager
├── requirements.txt                  # Consolidated Python dependencies
├── pixel-palette-extractor/          # Gradio app for extracting color palettes
├── tictactoe_tree/                   # Streamlit app for game AI trees
├── cellular-automata-classifier/     # Streamlit app for pattern classification
├── AI_Projects_Deployable/           # Deployable machine learning modules
├── refactor_script.py                # Refactoring workflow helper
└── LICENSE                           # MIT License
```

## Setup & Running Demos

### Prerequisites
- Python 3.8 or higher installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jalpan04/ML-playground.git
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Hub
Start the main dashboard:
```bash
streamlit run hub.py
```
Open `http://localhost:8501` in your browser. From here, you can start, stop, and access any of the individual demo applications with a single click.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
