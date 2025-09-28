import os
import shutil
import re

# --- Configuration ---
# This script assumes it's located in the 'ML projects' root directory.
# It will create a new directory for the deployable app.

OLD_PROJECT_ROOT = os.getcwd()
NEW_PROJECT_ROOT = os.path.join(OLD_PROJECT_ROOT, "AI_Projects_Deployable")

# Define source paths based on the original structure (removed decision_tree.png)
PATHS = {
    "cellular_app": "cellular-automata-classifier/app_cell.py",
    "cellular_models": "cellular-automata-classifier/saved_model",
    "cellular_helpers": [
        "cellular-automata-classifier/game_of_life.py",
        "cellular-automata-classifier/feature_extraction.py",
        "cellular-automata-classifier/patterns.py"
    ],
    "tictactoe_app": "tictactoe_tree/app_tic.py",
    "tictactoe_model": "tictactoe_tree/tictactoe_model.joblib",
    "palette_app": "pixel-palette-extractor/app_pix.py",
}

def main():
    """Main function to run the refactoring process."""
    print("🚀 Starting project refactoring for Streamlit deployment...")

    create_new_structure()
    process_cellular_automata()
    process_tictactoe()
    process_palette_extractor()
    create_home_page()
    create_requirements_file()
    create_gitignore_file()

    print("\n✅ Refactoring complete!")
    print(f"Your new deployable project is located in: '{NEW_PROJECT_ROOT}'")
    print("\nNext Steps:")
    print("1. Navigate into the new directory: cd AI_Projects_Deployable")
    print("2. Test it locally: streamlit run Home.py")
    print("3. Push this new folder to a GitHub repository to deploy.")

def create_new_structure():
    """Creates the necessary directories for the new app."""
    print("\n[Step 1/5] Creating new directory structure...")
    if os.path.exists(NEW_PROJECT_ROOT):
        print(f"  - Directory '{NEW_PROJECT_ROOT}' already exists. Clearing it.")
        shutil.rmtree(NEW_PROJECT_ROOT)
    os.makedirs(os.path.join(NEW_PROJECT_ROOT, "pages"))
    os.makedirs(os.path.join(NEW_PROJECT_ROOT, "assets"))
    os.makedirs(os.path.join(NEW_PROJECT_ROOT, "helpers"))
    print("  - Created: AI_Projects_Deployable, /pages, /assets, /helpers")

def process_cellular_automata():
    """Copies and refactors the Cellular Automata app."""
    print("\n[Step 2/5] Processing: Cellular Automata Classifier...")
    for helper_path in PATHS["cellular_helpers"]:
        shutil.copy(helper_path, os.path.join(NEW_PROJECT_ROOT, "helpers"))
    print("  - Copied helper scripts to /helpers.")
    shutil.copytree(PATHS["cellular_models"], os.path.join(NEW_PROJECT_ROOT, "assets", "cellular_automata_models"))
    print("  - Copied models to /assets/cellular_automata_models.")
    new_app_path = os.path.join(NEW_PROJECT_ROOT, "pages", "3_🧬_Cellular_Automata.py")
    shutil.copy(PATHS["cellular_app"], new_app_path)
    with open(new_app_path, "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        new_header = "import sys\nimport os\nsys.path.append('helpers')\n\n" + content
        updated_content = re.sub(
            r'os\.path\.join\("saved_model",\s*"([^"]+)"\)',
            r'os.path.join("assets", "cellular_automata_models", "\1")', new_header
        )
        f.write(updated_content)
        f.truncate()
    print("  - Copied and updated app file with new asset paths.")

def process_tictactoe():
    """Copies and refactors the Tic-Tac-Toe app, removing the decision tree UI."""
    print("\n[Step 3/5] Processing: Tic-Tac-Toe AI...")
    shutil.copy(PATHS["tictactoe_model"], os.path.join(NEW_PROJECT_ROOT, "assets"))
    print("  - Copied model to /assets.")

    new_app_path = os.path.join(NEW_PROJECT_ROOT, "pages", "2_Tictactoe_AI.py")
    shutil.copy(PATHS["tictactoe_app"], new_app_path)

    # This new main function removes the two-column layout and the image display.
    # It also adds st.rerun() for a more responsive UI.
    new_main_function_code = r'''
def main():
    st.set_page_config(layout="wide", page_title="Unbeatable Tic-Tac-Toe AI")

    # Initialize session state
    if "board" not in st.session_state:
        st.session_state.board = np.zeros((3, 3), int)
        st.session_state.player_turn = True
        st.session_state.game_over = False
        st.session_state.winner_info = None
        st.session_state.last_click = None

    st.write("## Unbeatable Tic-Tac-Toe AI")
    st.write("You are 'O' (Red). The AI is 'X' (Teal).")

    st.subheader("Game Board")

    if st.button("New Game"):
        # Clear the entire session state for a clean reset
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Game Status
    if st.session_state.get('game_over', False):
        winner = st.session_state.winner_info[0]
        if winner == 1: st.error("You lost! AI wins.")
        elif winner == -1: st.success("You won! Congrats.")
        else: st.warning("It's a draw!")
    elif st.session_state.get('player_turn', True):
        st.info("Your turn. Click a square.")
    else:
        st.info("AI is thinking...")

    # Draw Board
    board_img = draw_board(st.session_state.get('board', np.zeros((3, 3), int)), st.session_state.get('winner_info'))
    click = streamlit_image_coordinates(board_img, key="board_click")
    st.caption("AI uses a Decision Tree model with a Minimax fallback.")

    # --- Game Logic ---
    # Player Move
    if st.session_state.get('player_turn', True) and not st.session_state.get('game_over', False):
        if click and click != st.session_state.get('last_click'):
            st.session_state.last_click = click
            row, col = click['y'] // SQUARE_SIZE, click['x'] // SQUARE_SIZE
            if st.session_state.board[row, col] == 0:
                st.session_state.board[row, col] = -1
                st.session_state.player_turn = False
                st.rerun()

    # Check Winner
    if not st.session_state.get('game_over', False):
        winner_info = check_winner(st.session_state.board)
        if winner_info:
            st.session_state.game_over = True
            st.session_state.winner_info = winner_info
            st.rerun()

    # AI Move
    if not st.session_state.get('player_turn', True) and not st.session_state.get('game_over', False):
        with st.spinner("AI is thinking..."):
            time.sleep(0.5)
        board_flat = st.session_state.board.flatten()
        ai_move = AI_MODEL.predict([board_flat])[0]
        r, c = divmod(ai_move, 3)
        if st.session_state.board[r, c] != 0:
            ai_move = best_move_minimax(board_flat)
            r, c = divmod(ai_move, 3)
        st.session_state.board[r, c] = 1
        st.session_state.player_turn = True
        st.rerun()
'''
    with open(new_app_path, "r+", encoding="utf-8") as f:
        content = f.read()
        # 1. Fix MODEL_PATH variable at the top of the file
        content = re.sub(
            r"MODEL_PATH\s*=\s*['\"]tictactoe_model\.joblib['\"]",
            "MODEL_PATH = 'assets/tictactoe_model.joblib'",
            content
        )
        # 2. Replace the entire old main function with the new, cleaner version
        content = re.sub(
            r'def main\(\):.*?(?=if __name__ == "__main__":)',
            new_main_function_code,
            content,
            flags=re.DOTALL
        )
        f.seek(0)
        f.write(content)
        f.truncate()
    print("  - Copied and refactored app, removing decision tree UI.")


def process_palette_extractor():
    """Converts the Gradio Palette Extractor to a Streamlit app."""
    print("\n[Step 4/5] Processing: Pixel Palette Extractor (and converting to Streamlit)...")
    new_app_path = os.path.join(NEW_PROJECT_ROOT, "pages", "1_🎨_Palette_Extractor.py")
    streamlit_code = """
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import io

def create_palette_image(colors, swatch_size=50, spacing=10):
    num_colors = len(colors)
    width = (swatch_size * num_colors) + (spacing * (num_colors + 1))
    height = swatch_size + (2 * spacing)
    palette = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(palette)
    for i, color in enumerate(colors):
        x0, y0 = spacing + i * (swatch_size + spacing), spacing
        x1, y1 = x0 + swatch_size, y0 + swatch_size
        draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline=(50, 50, 50))
    return palette

def extract_palette(input_image, num_colors):
    if not input_image: return None
    image = input_image.resize((200, 200))
    np_image = np.array(image)
    pixels = np_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=int(num_colors), random_state=42, n_init='auto')
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return create_palette_image(dominant_colors)

st.set_page_config(layout="wide", page_title="Pixel Palette Extractor")
st.title("🎨 Pixel Palette Extractor")
st.markdown("Upload an image to extract its dominant color palette using **K-Means Clustering**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
num_colors = st.slider("Number of Colors", 2, 20, 8)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Generate Palette", type="primary", use_container_width=True):
        with st.spinner("Extracting colors..."):
            palette = extract_palette(input_image, num_colors)
            with col2: st.image(palette, caption=f"Generated {num_colors}-Color Palette")
else:
    st.info("Upload an image to get started.")
"""
    with open(new_app_path, "w", encoding="utf-8") as f:
        f.write(streamlit_code)
    print("  - Created new Streamlit version of the app.")

def create_home_page():
    """Creates the main landing page for the app."""
    print("\n[Step 5/5] Creating supplementary files...")
    home_page_code = """
import streamlit as st

st.set_page_config(
    page_title="ML Projects Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to the AI Projects Hub! 👋")
st.write("A central showcase of machine learning projects. Use the sidebar to navigate to each application.")
st.divider()

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.header("🎨 Palette Extractor")
    st.image("https://i.imgur.com/Sd1eE9j.png")
    st.write("Upload an image to extract its dominant color palette using **K-Means Clustering**.")

with col2:
    st.header(" Tic-Tac-Toe AI")
    st.image("https://i.imgur.com/L7sT3hV.png")
    st.write("Play against an unbeatable AI that uses a **Decision Tree** and the **Minimax algorithm**.")

with col3:
    st.header("🧬 Cellular Automata")
    st.image("https://i.imgur.com/a8hA37J.gif")
    st.write("A simulation of Conway's Game of Life that uses **K-Means Clustering** to classify emerging patterns in real-time.")

st.sidebar.success("Select an application above.")
"""
    with open(os.path.join(NEW_PROJECT_ROOT, "Home.py"), "w", encoding="utf-8") as f:
        f.write(home_page_code)
    print("  - Created Home.py landing page.")

def create_requirements_file():
    """Creates a consolidated requirements.txt file."""
    requirements = ["streamlit", "numpy", "scikit-learn", "Pillow", "pygame", "matplotlib", "joblib", "streamlit-image-coordinates"]
    with open(os.path.join(NEW_PROJECT_ROOT, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(requirements))
    print("  - Created requirements.txt file.")

def create_gitignore_file():
    """Creates a .gitignore file."""
    gitignore_content = "*.pyc\n__pycache__/\n.DS_Store\n.venv/"
    with open(os.path.join(NEW_PROJECT_ROOT, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("  - Created .gitignore file.")

if __name__ == "__main__":
    main()

