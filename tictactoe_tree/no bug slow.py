# app_fixed_proper.py

import streamlit as st
import pygame
import numpy as np
import joblib
import time
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Constants & Setup ---
SCREEN_WIDTH, SCREEN_HEIGHT = 300, 300
SQUARE_SIZE = SCREEN_WIDTH // 3
MODEL_PATH = 'tictactoe_model.joblib'

# --- Visuals (Dark Mode Palette) ---
BG_COLOR = (20, 20, 20)
LINE_COLOR = (80, 80, 80)
X_COLOR = (0, 200, 200)  # Teal
O_COLOR = (255, 120, 120)  # Light Red
WIN_COLOR = (255, 215, 0)  # Gold
GRID_LINE_WIDTH, UNIFIED_WIDTH = 8, 12
CIRCLE_RADIUS, MARGIN = SQUARE_SIZE // 3, 30

# --- AI and Game Logic ---
try:
    AI_MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please run train_model_fixed.py first.")
    st.stop()


def check_winner(board):
    """Check winner and return (winner, type, index) or None."""
    # Rows
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return (board[i, 0], "row", i)
    # Columns
    for i in range(3):
        if board[0, i] == board[1, i] == board[2, i] != 0:
            return (board[0, i], "col", i)
    # Diagonals
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return (board[0, 0], "diag", 1)
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return (board[0, 2], "diag", 2)
    # Draw
    if 0 not in board.flatten():
        return (0, "draw", None)
    return None


# --- Minimax Fallback ---
def minimax(board_flat, is_maximizing):
    winner = check_winner(np.array(board_flat).reshape(3, 3))
    if winner:
        val = winner[0]
        if val == 1: return 10
        if val == -1: return -10
        return 0

    if is_maximizing:
        best_score = -np.inf
        for i in range(9):
            if board_flat[i] == 0:
                board_flat[i] = 1
                score = minimax(board_flat, False)
                board_flat[i] = 0
                best_score = max(best_score, score)
        return best_score
    else:
        best_score = np.inf
        for i in range(9):
            if board_flat[i] == 0:
                board_flat[i] = -1
                score = minimax(board_flat, True)
                board_flat[i] = 0
                best_score = min(best_score, score)
        return best_score


def best_move_minimax(board_flat):
    best_score = -np.inf
    move = -1
    for i in range(9):
        if board_flat[i] == 0:
            board_flat[i] = 1
            score = minimax(board_flat, False)
            board_flat[i] = 0
            if score > best_score:
                best_score = score
                move = i
    return move


# --- Pygame Drawing ---
def draw_board(board, winner_info):
    pygame.init()
    screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(BG_COLOR)
    # Grid
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (SCREEN_WIDTH, i * SQUARE_SIZE), GRID_LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, SCREEN_HEIGHT), GRID_LINE_WIDTH)
    # Pieces
    for r in range(3):
        for c in range(3):
            center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
            if board[r, c] == -1:
                pygame.draw.circle(screen, O_COLOR, center, CIRCLE_RADIUS, UNIFIED_WIDTH)
            elif board[r, c] == 1:
                pygame.draw.line(screen, X_COLOR,
                                 (c * SQUARE_SIZE + MARGIN, r * SQUARE_SIZE + MARGIN),
                                 (c * SQUARE_SIZE + SQUARE_SIZE - MARGIN, r * SQUARE_SIZE + SQUARE_SIZE - MARGIN),
                                 UNIFIED_WIDTH)
                pygame.draw.line(screen, X_COLOR,
                                 (c * SQUARE_SIZE + SQUARE_SIZE - MARGIN, r * SQUARE_SIZE + MARGIN),
                                 (c * SQUARE_SIZE + MARGIN, r * SQUARE_SIZE + SQUARE_SIZE - MARGIN),
                                 UNIFIED_WIDTH)
    # Winning line
    if winner_info and winner_info[0] != 0:
        typ, idx = winner_info[1], winner_info[2]
        if typ == "row":
            y = idx * SQUARE_SIZE + SQUARE_SIZE / 2
            pygame.draw.line(screen, WIN_COLOR, (15, y), (SCREEN_WIDTH - 15, y), 15)
        elif typ == "col":
            x = idx * SQUARE_SIZE + SQUARE_SIZE / 2
            pygame.draw.line(screen, WIN_COLOR, (x, 15), (x, SCREEN_HEIGHT - 15), 15)
        elif typ == "diag":
            if idx == 1:
                pygame.draw.line(screen, WIN_COLOR, (25, 25), (SCREEN_WIDTH - 25, SCREEN_HEIGHT - 25), 15)
            else:
                pygame.draw.line(screen, WIN_COLOR, (25, SCREEN_HEIGHT - 25), (SCREEN_WIDTH - 25, 25), 15)

    arr = pygame.surfarray.array3d(screen)
    pygame.quit()
    return np.transpose(arr, (1, 0, 2))


# --- Main App ---
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

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Game Board")

        # --- New Game Button ---
        if st.button("New Game"):
            st.session_state.board = np.zeros((3, 3), int)
            st.session_state.player_turn = True
            st.session_state.game_over = False
            st.session_state.winner_info = None
            st.session_state.last_click = None
            if "board_click" in st.session_state:
                del st.session_state["board_click"]

        # --- Game Status ---
        if st.session_state.game_over:
            winner = st.session_state.winner_info[0]
            if winner == 1:
                st.error("You lost! AI wins.")
            elif winner == -1:
                st.success("You won! Congrats.")
            else:
                st.warning("It's a draw!")
        elif st.session_state.player_turn:
            st.info("Your turn. Click a square.")
        else:
            st.info("AI is thinking...")

        # --- Draw Board ---
        board_img = draw_board(st.session_state.board, st.session_state.winner_info)
        click = streamlit_image_coordinates(board_img, key="board_click")

    # --- Player Move ---
    if st.session_state.player_turn and not st.session_state.game_over:
        if click and click != st.session_state.last_click:
            st.session_state.last_click = click
            row, col = click['y'] // SQUARE_SIZE, click['x'] // SQUARE_SIZE
            if st.session_state.board[row, col] == 0:
                st.session_state.board[row, col] = -1
                st.session_state.player_turn = False

    # --- Check Winner ---
    if not st.session_state.game_over:
        winner_info = check_winner(st.session_state.board)
        if winner_info:
            st.session_state.game_over = True
            st.session_state.winner_info = winner_info

    # --- AI Move ---
    if not st.session_state.player_turn and not st.session_state.game_over:
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

    with col2:
        st.subheader("AI's Brain")
        st.image("decision_tree.png")
        st.caption("AI uses a Decision Tree first. If invalid, it falls back to Minimax.")


if __name__ == "__main__":
    main()