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
    BOARD_SIZE=7,
    REPLAY_BUFFER_CAPACITY=20000,
    EVAL_SIMS = 800,
    NUM_GAMES_EVAL = 40,
    CURRENT_SIMS = 400,
    CHECKPOINT_EVAL_DIR = os.path.join(os.path.dirname(train.__file__), "checkpoints_eval_small"),

    # --- Temperature Schedule (smaller board = fewer turns, less exploration) ---
    TEMP_HIGH_TURNS=2,
    TEMP_MID_TURNS=0,
    TEMP_LOW = 0.1,

    # --- Paths (separate from the 11x11 model) ---
    CHECKPOINT_DIR=os.path.join(os.path.dirname(train.__file__), "checkpoints_small"),
    BUFFER_DIR=os.path.join(os.path.dirname(train.__file__), "buffers_small"),
    LOG_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "training_small.log"),
    EVAL_LOG_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "evals_small.log"),
    SET_EVAL_STATE_FILE=os.path.join(os.path.dirname(train.__file__), "logs", "set_eval_state_small.json"),
    
    # --- Eval Config ---
    SET_EVAL_EVERY = 10,
    SET_CHECKPOINT = 20
)

if __name__ == "__main__":
    train.run_training()
