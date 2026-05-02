"""
Train a smaller (5x5) Hex model.

This script reuses ALL training logic from train.py, overriding only the
hyperparameters and paths that differ for the smaller board.  When you change
how training works, you only need to edit train.py — this file stays untouched.
"""
import agents.cem.train as train

train.configure(
    # --- Board ---
    BOARD_SIZE=5,

    # --- Temperature Schedule (smaller board = fewer turns, less exploration) ---
    TEMP_HIGH_TURNS=4,
    TEMP_MID_TURNS=0,

    # --- Paths (separate from the 11x11 model) ---
    CHECKPOINT_DIR="agents/cem/checkpoints_small/",
    BUFFER_DIR="agents/cem/buffers_small/",
    LOG_FILE="agents/cem/logs/training_small.log",
    EVAL_LOG_FILE = "agents/cem/logs/evals_small.log"
)

if __name__ == "__main__":
    train.run_training()
