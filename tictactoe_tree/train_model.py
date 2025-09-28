# train_model_fixed.py

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from collections import deque


# --- Core Game Logic (Minimax) ---

def check_winner(board):
    """
    Checks for a winner on the board.
    Returns: 1 for AI (X), -1 for Player (O), 0 for Draw, None for ongoing game.
    """
    board = np.array(board).reshape(3, 3)
    # Check rows and columns
    for i in range(3):
        if abs(sum(board[i, :])) == 3: return board[i, 0]
        if abs(sum(board[:, i])) == 3: return board[0, i]
    # Check diagonals
    if abs(sum(np.diag(board))) == 3: return board[0, 0]
    if abs(sum(np.diag(np.fliplr(board)))) == 3: return board[0, 2]
    # Check for draw (no empty squares left)
    if 0 not in board: return 0
    return None  # Game is still ongoing


def minimax(board, depth, is_maximizing):
    """Minimax algorithm to determine the best score for a given board state."""
    winner = check_winner(board)
    if winner is not None:
        if winner == 1: return 10 - depth
        if winner == -1: return depth - 10
        if winner == 0: return 0

    if is_maximizing:
        best_score = -np.inf
        for move in range(9):
            if board[move] == 0:
                board[move] = 1
                score = minimax(board, depth + 1, False)
                board[move] = 0
                best_score = max(score, best_score)
        return best_score
    else:  # Minimizing player
        best_score = np.inf
        for move in range(9):
            if board[move] == 0:
                board[move] = -1
                score = minimax(board, depth + 1, True)
                board[move] = 0
                best_score = min(score, best_score)
        return best_score


def find_best_move(board):
    """Finds the best move for the AI (player 1) using minimax."""
    best_score = -np.inf
    best_move = -1
    for move in range(9):
        if board[move] == 0:
            board[move] = 1  # AI's move
            score = minimax(board, 0, False)
            board[move] = 0  # Undo move
            if score > best_score:
                best_score = score
                best_move = move
    return best_move


# --- Data Generation ---

def generate_data():
    """Generates all possible board states and their optimal moves."""
    print("Generating training data... This may take a moment.")
    game_states = []
    q = deque([([0] * 9, 1)])  # (board, next_player)
    visited = {tuple([0] * 9)}

    while q:
        board, player = q.popleft()

        # *** BUG FIX ***
        # Only find a best move if the game is still ongoing.
        # This prevents adding data for boards that are already finished.
        current_winner = check_winner(board)
        if current_winner is not None:
            continue

        # If it's AI's turn, find and record the best move
        if player == 1:
            best_move = find_best_move(board)
            if best_move != -1:
                game_states.append(board + [best_move])

        # Generate next states
        for move in range(9):
            if board[move] == 0:
                next_board = list(board)
                next_board[move] = player
                if tuple(next_board) not in visited:
                    q.append((next_board, -player))
                    visited.add(tuple(next_board))

    # Create a DataFrame
    columns = [f'square_{i}' for i in range(9)] + ['best_move']
    df = pd.DataFrame(game_states, columns=columns)
    df.drop_duplicates(subset=columns[:-1], inplace=True)
    df.to_csv('tictactoe_data.csv', index=False)
    print(f"Data generation complete. Found {len(df)} unique game states for the AI to learn from.")
    return df


# --- Model Training and Saving ---

def train_and_save_model():
    """Loads data, trains the model, and saves artifacts."""
    try:
        df = pd.read_csv('tictactoe_data.csv')
        print("Loaded existing training data from 'tictactoe_data.csv'.")
    except FileNotFoundError:
        print("No existing data found.")
        df = generate_data()

    X = df.iloc[:, :9]
    y = df.iloc[:, 9]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree
    # A max_depth of 9 is sufficient for Tic-Tac-Toe, as there are 9 moves.
    model = DecisionTreeClassifier(max_depth=9, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    # Save the model
    joblib.dump(model, 'tictactoe_model.joblib')
    print("Model saved to tictactoe_model.joblib")

    # Visualize the tree using matplotlib
    print("Generating tree visualization with matplotlib...")
    plt.figure(figsize=(20, 10), dpi=300)

    plot_tree(model,
              feature_names=[f'square_{i}' for i in range(9)],
              class_names=[f'Move {i}' for i in range(9)],
              filled=True,
              rounded=True,
              max_depth=4,  # Limit visual depth for readability
              fontsize=6)

    plt.title("Decision Tree for Tic-Tac-Toe AI (Partial View)")
    plt.savefig('decision_tree.png')
    print("Decision tree visualization saved to decision_tree.png")


if __name__ == "__main__":
    train_and_save_model()