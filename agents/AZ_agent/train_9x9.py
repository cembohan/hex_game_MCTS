"""
Train a smaller (5x5) Hex model.

This script reuses ALL training logic from train.py, overriding only the
hyperparameters and paths that differ for the smaller board.  When you change
how training works, you only need to edit train.py — this file stays untouched.
"""
import os
import agents.AZ_agent.train as train

train.configure(
    # --- Board ---
    BOARD_SIZE=9,
    REPLAY_BUFFER_CAPACITY=30000,
    EVAL_SIMS = 1000,
    NUM_GAMES_EVAL = 40,
    EVAL_EVERY = 1000,
    SET_EVAL_EVERY = 1000,
    CURRENT_SIMS = 600,
    CHECKPOINT_EVAL_DIR = os.path.join(os.path.dirname(train.__file__), "checkpoints_eval_9x9"),

    # --- Temperature Schedule (smaller board = fewer turns, less exploration) ---
    TEMP_HIGH_TURNS=5,
    TEMP_MID_TURNS=0,
    TEMP_LOW = 0.15,

    # --- Paths (separate from the 11x11 model) ---
    CHECKPOINT_DIR=os.path.join(os.path.dirname(train.__file__), "checkpoints_9x9"),
    BUFFER_DIR=os.path.join(os.path.dirname(train.__file__), "buffers_9x9"),
    LOG_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "training_9x9.log"),
    EVAL_LOG_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "evals_9x9.log"),
    SET_EVAL_STATE_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "set_eval_state_9x9.json"),
    
    # --- Eval Config ---
    SET_CHECKPOINT = 10
)

if __name__ == "__main__":
    train.run_training()
