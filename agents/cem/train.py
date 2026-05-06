import os
import random
import signal
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import gc
from collections import deque

# Graceful termination flag
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle Ctrl+C and termination signals gracefully."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print(f"\n>>> Received {sig_name}. Finishing current step, then saving and exiting...", flush=True)
    _shutdown_requested = True

def _check_shutdown():
    """Check if graceful shutdown was requested."""
    return _shutdown_requested

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.cem.agent1 import HexPVNet
from agents.cem.mcts import MCTS, BatchedMCTS, Node, encode_state, get_valid_moves, apply_move, init_board_config

import logging
#TODO - Add notifications on eval completion (discord, desktop etc.)
#TODO - (consider) Occasionally start the model from a random node in the MCTS tree instead of the root, to expose it to a wider variety of positions and avoid overfitting to the early game.
"""TODO - (ASAP) after the robust implementation of swap rule, keep an eye on the eval cycles,
if the eval isn't passing (which is now less likely due to newly introduced strategy space),
delete the buffer.pt and potentially lower the eval win rate threshold.
"""

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# --- Replay Buffer ---
REPLAY_BUFFER_CAPACITY = 100000  # Max games to store in buffer

# --- Training (Optimizer) ---
LEARNING_RATE = 5e-4           # Adam optimizer learning rate
WEIGHT_DECAY = 1e-4             # L2 regularization strength

# --- Training Loop ---
EPOCHS = 3000                   # Total training epochs
GAMES_PER_EPOCH = 24            # Self-play games per epoch
BATCH_SIZE = 256                # Training batch size
TRAINING_STEPS = 200            # Gradient updates per epoch
EVAL_EVERY = 7                 # Evaluate every N epochs
NUM_GAMES_EVAL = 60             # Games for evaluation
EXPLORATORY_EVERY = 10          # Self-play games per epoch

# --- MCTS Simulations ---
# --- Dynamic Sims Settings ---
CURRENT_SIMS = 175
MIN_SIMS = 100
SIMS_DECAY_AMOUNT = 5
EVALS_PASSED = 0  # Counter to track progress
EVAL_SIMS = 125                # MCTS simulations per move during evaluation
OPPONENT_GAME_SIMS = 100       # MCTS simulations per move vs local agents

# --- MCTS Settings ---
MCTS_TEMPERATURE = 1.0          # Exploration temperature for self-play
MCTS_TEMPERATURE_EVAL = 0.3     # Exploration temperature for evaluation
ADD_NOISE = True                # Add Dirichlet noise to root (self-play only)
C_PUCT = 2.0                   # UCB exploration constant (unified for MCTS and BatchedMCTS)
MAX_EXPANSION_WIDTH = False     # Toggle for top-k node expansion. Set to an int (e.g., 16) for top-k, or False/None for full regular MCTS expansion

# --- Temperature Schedule (for move selection) ---
TEMP_HIGH_TURNS = 12            # Use high temp for first N turns
TEMP_MID_TURNS = 0             # Use mid temp for next N turns
TEMP_HIGH = 1.0                 # High temperature value
TEMP_MID = 0.7                  # Medium temperature value
TEMP_LOW = 0.2                  # Low temperature value (greedy)

# --- Opponent Diversity ---
OPPONENT_GAME_EVERY = 0         # Play vs local agent every N games (0 to disable)

# --- Evaluation ---
EVAL_WIN_RATE_THRESHOLD = 0.51  # Win rate needed to replace best model (55%)

# --- Board Size (fixed for now) ---
BOARD_SIZE = 11

# --- Folder Paths ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_BASE_DIR, "checkpoints")
BUFFER_DIR = os.path.join(_BASE_DIR, "buffers")
LOG_FILE = os.path.join(_BASE_DIR, "logs", "training.log")
EVAL_LOG_FILE = os.path.join(_BASE_DIR, "logs", "evals.log")

# Logger is set up lazily via _setup_logging() so that configure() can
# change LOG_FILE / CHECKPOINT_DIR / BUFFER_DIR before anything runs.
logger = logging.getLogger("HexTraining")
# Dedicated logger for evaluation results — routed to a separate file.
eval_logger = logging.getLogger("HexEval")


def configure(**overrides):
    """Override any module-level constant before training starts.

    Example::
        import agents.cem.train as train
        train.configure(BOARD_SIZE=5, CHECKPOINT_DIR="agents/cem/checkpoints_small/")
        train.run_training()
    """
    g = globals()
    for key, value in overrides.items():
        if key not in g:
            raise KeyError(f"Unknown config key: {key}")
        g[key] = value


def _setup_logging():
    """Initialise board config, directories, and logging handlers.

    Must be called once *after* any configure() call.
    """
    init_board_config(BOARD_SIZE)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    formatter.converter = time.localtime

    # --- Main training logger (file + console) ---
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers when called more than once
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # --- Dedicated eval logger (file only, no console spam) ---
    eval_logger.setLevel(logging.INFO)
    # Prevent propagation so eval entries don't double-write to root handlers
    eval_logger.propagate = False
    if not eval_logger.handlers:
        eval_fh = logging.FileHandler(EVAL_LOG_FILE)
        eval_fh.setFormatter(formatter)
        eval_logger.addHandler(eval_fh)

import pickle # Ensure this is imported at the top

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
        
    def save_game(self, game_history, winner_colour):
        for state, pi, current_colour, action, valid_moves in game_history:
            z = 1.0 if current_colour == winner_colour else -1.0
            
            # 1. Convert state to numpy to avoid massive PyTorch storage overhead
            state_np = state.numpy() if isinstance(state, torch.Tensor) else state
            
            self.buffer.append((state_np, pi, z, action, valid_moves))
            
            # Data Augmentation: Hex 180-degree rotation using NumPy
            rotated_state_np = np.rot90(state_np, 2, axes=(2, 3)).copy()
            
            bs = BOARD_SIZE
            swap_idx = bs * bs

            pi_board = pi[:-1].reshape(bs, bs)
            rotated_pi_board = np.rot90(pi_board, 2)
            rotated_pi = np.append(rotated_pi_board.flatten(), pi[-1]) 
            
            if action != swap_idx:
                x, y = divmod(action, bs)
                rx, ry = bs - 1 - x, bs - 1 - y
                rotated_action = rx * bs + ry
            else:
                rotated_action = swap_idx
                
            rotated_valid_moves = []
            for vm in valid_moves:
                if vm != swap_idx:
                    vx, vy = divmod(vm, bs)
                    rx, ry = bs - 1 - vx, bs - 1 - vy
                    rotated_valid_moves.append(rx * bs + ry)
                else:
                    rotated_valid_moves.append(swap_idx)
                    
            self.buffer.append((rotated_state_np, rotated_pi, z, rotated_action, rotated_valid_moves))
            
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty.")

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, pis, zs, actions, valid_moves_list = zip(*batch)
        
        states_tensor = torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32)
        pis_tensor = torch.tensor(np.array(pis), dtype=torch.float32)
        zs_tensor = torch.tensor(zs, dtype=torch.float32)
        
        return states_tensor, pis_tensor, zs_tensor, actions, valid_moves_list
        
    def __len__(self):
        return len(self.buffer)
        
    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(BUFFER_DIR, "buffer.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 3. Use standard pickle for lists of standard python/numpy objects
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        
    def load(self, filename=None):
        if filename is None:
            filename = os.path.join(BUFFER_DIR, "buffer.pt")
        if not os.path.isfile(filename):
            print(f"No replay buffer found at {filename}. Starting with an empty buffer.")
            return

        loaded = None
        load_error = None
        try:
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)
                print(f"Loaded {len(loaded)} entries from pickle buffer file {filename}.")
        except Exception as exc:
            load_error = exc
            try:
                loaded = torch.load(filename, map_location="cpu", weights_only=False)
                print(f"Loaded {len(loaded)} entries from torch buffer file {filename}.")
            except Exception as exc2:
                print(f"Warning: failed to load replay buffer from {filename}.\n"
                      f"  pickle error: {load_error}\n"
                      f"  torch error: {exc2}\n"
                      "Starting with an empty buffer.")
                self.buffer = deque(maxlen=self.buffer.maxlen)
                return

        cleaned_loaded = []
        for item in loaded:
            if len(item) == 6:
                state, pi, _, z, action, valid_moves = item
            else:
                state, pi, z, action, valid_moves = item
            state_np = state.numpy() if isinstance(state, torch.Tensor) else state
            cleaned_loaded.append((state_np, pi, z, action, valid_moves))

        self.buffer = deque(cleaned_loaded, maxlen=self.buffer.maxlen)
        print(f"Loaded {len(self.buffer)} games from {filename} and normalized states.")


class HexTrainer:
    def __init__(self, model, lr=LEARNING_RATE):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.history = {'loss': [], 'v_loss': [], 'p_loss': []}

    def train_step(self, states, target_pis, target_vs, valid_moves_list):
        self.model.train()
        device = next(self.model.parameters()).device
        states = states.to(device)
        target_pis = target_pis.to(device)
        target_vs = target_vs.to(device)

        self.optimizer.zero_grad()

        # Forward pass
        p_logits, v = self.model(states)

        # 1. Masking Invalid Moves in Policy
        # Create a boolean mask of the same shape as p_logits (Batch, BOARD_SIZE*BOARD_SIZE+1)
        mask = torch.ones_like(p_logits, dtype=torch.bool)
        for i, valid in enumerate(valid_moves_list):
            mask[i, valid] = False # False means it IS a valid move
            
        # Overwrite illegal logits with a massive negative number
        p_logits = p_logits.masked_fill(mask, -1e9)

        # Now the softmax will perfectly ignore illegal moves
        p_loss = -torch.mean(torch.sum(target_pis * F.log_softmax(p_logits, dim=1), dim=1))

        # 2. State-Value Loss (MSE)
        v_loss = F.mse_loss(v.view(-1), target_vs)

        total_loss = p_loss + v_loss
        total_loss.backward()
        self.optimizer.step()

        self.history['loss'].append(total_loss.item())
        self.history['p_loss'].append(p_loss.item())
        self.history['v_loss'].append(v_loss.item())

        return total_loss.item(), p_loss.item(), v_loss.item()

    def save_checkpoint(self, iteration, path=None):
        if path is None:
            path = CHECKPOINT_DIR
        os.makedirs(path, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'board_size': BOARD_SIZE,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path, f"checkpoint_{iteration}.pt"))
        
    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        if os.path.isfile(filename):
            device = next(self.model.parameters()).device
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('iteration', 0)
        return 0
    
# ---------------------------------------------------------------------------
# Local-agent pool for opponent-diversity games
# ---------------------------------------------------------------------------
# Add or remove entries here to change which agents can be drawn as opponents.
# Each entry is a (module_path, class_name) pair that will be imported lazily.
_LOCAL_AGENT_POOL = [
    ("agents.DefaultAgents.NaiveAgent", "NaiveAgent"),
    ("agents.TestAgents.ValidAgent",    "ValidAgent"),
    ("agents.Group997.NaiveAgent",      "NaiveAgent"),
    ("agents.Group998.NaiveAgent",      "NaiveAgent"),
    ("agents.Group999.NaiveAgent",      "NaiveAgent"),
]


def load_random_local_agent(colour: Colour):
    """Randomly pick one agent class from _LOCAL_AGENT_POOL and instantiate it."""
    import importlib
    module_path, class_name = random.choice(_LOCAL_AGENT_POOL)
    try:
        module = importlib.import_module(module_path)
        AgentClass = getattr(module, class_name)
        return AgentClass(colour)
    except Exception as e:
        logger.warning(f"Failed to load {module_path}.{class_name}: {e}. Falling back to DefaultAgents.NaiveAgent")
        from agents.DefaultAgents.NaiveAgent import NaiveAgent
        return NaiveAgent(colour)


def play_vs_agent(model, buffer, mcts_simulations=OPPONENT_GAME_SIMS):
    """
    Play a single game where the model (using MCTS) faces a local AgentBase opponent.
    The model is randomly assigned RED or BLUE each call.
    Returns the winner Colour (or None).
    """
    model.eval()
    board = Board(BOARD_SIZE)

    # Randomly decide which side the model plays
    model_colour = random.choice([Colour.RED, Colour.BLUE])
    opp_colour   = Colour.opposite(model_colour)

    # Instantiate the random opponent agent
    opp_agent = load_random_local_agent(opp_colour)
    logger.debug(f"Opponent-diversity game: model={model_colour}, opp={type(opp_agent).__name__}")

    mcts = MCTS(model, num_simulations=mcts_simulations, c_puct=C_PUCT, temperature=MCTS_TEMPERATURE, max_expansion_width=MAX_EXPANSION_WIDTH)

    history      = []          # (state_tensor, pi, colour, action, valid_moves)
    current_colour = Colour.RED
    turn           = 1
    last_move: Move | None = None
    winner         = None

    while True:
        if current_colour == model_colour:
            # ---- Model's turn: use MCTS ----
            # MCTS always runs at temperature=1.0 to get the raw visit distribution for the replay buffer.
            raw_pi_tensor = mcts.search(board, current_colour, turn)
            device = next(model.parameters()).device

            valid_moves = get_valid_moves(board, turn)

            # --- Temperature schedule for Action Selection ---
            temp = (TEMP_HIGH if turn < TEMP_HIGH_TURNS
                    else TEMP_MID if turn < TEMP_MID_TURNS
                    else TEMP_LOW)
            
            if temp != 1.0:
                sample_pi = raw_pi_tensor ** (1.0 / temp)
                sample_pi = sample_pi / sample_pi.sum()
            else:
                sample_pi = raw_pi_tensor

            action = int(torch.multinomial(sample_pi, 1).item())

            # Encode state for the neural network. (Note: The NN only needs to know 
            # if it's turn 2 for the swap rule, otherwise we pass turn=None).
            state_cpu = encode_state(board, current_colour, device,
                                     turn=turn if turn == 2 else None).cpu()
            
            # CRITICAL: We must store the RAW unsharpened pi in the history as the target!
            history.append((state_cpu, raw_pi_tensor.numpy(), current_colour, action, valid_moves))

            # Apply the model's move
            move_obj = Move(-1, -1) if action == BOARD_SIZE * BOARD_SIZE else Move(*divmod(action, BOARD_SIZE))
            new_board, next_col, next_turn, is_terminal, winner = apply_move(
                board, current_colour, turn, action
            )
            last_move = move_obj
        else:
            # ---- Opponent agent's turn ----
            move_obj = opp_agent.make_move(turn, board, last_move)

            if move_obj.x == -1 and move_obj.y == -1:
                # Swap move
                action = BOARD_SIZE * BOARD_SIZE
            else:
                action = move_obj.x * BOARD_SIZE + move_obj.y

            # Validate: if the move lands on an occupied cell, skip (play first valid)
            if action != BOARD_SIZE * BOARD_SIZE and action not in get_valid_moves(board, turn):
                valid = get_valid_moves(board, turn)
                if not valid:
                    break  # Should not happen, but guard
                action = valid[0]
                move_obj = Move(*divmod(action, BOARD_SIZE))

            new_board, next_col, next_turn, is_terminal, winner = apply_move(
                board, current_colour, turn, action
            )
            last_move = move_obj

        board          = new_board
        current_colour = next_col
        turn           = next_turn

        if is_terminal:
            break

        # Safety valve: game should never last this long
        max_turns = BOARD_SIZE * BOARD_SIZE + 4
        if turn > max_turns:
            logger.warning(f"play_vs_agent: hit turn limit {max_turns} without terminal — forcing stop")
            break

    buffer.save_game(history, winner)
    return winner


def recursive_free(node):
    if node is None:
        return
    
    # 1. Break parent reference
    node.parent = None
    
    # 2. Recursively free all children
    if hasattr(node, 'children_nodes') and node.children_nodes is not None:
        for child in node.children_nodes:
            if child is not None:
                recursive_free(child)
                
    # 3. Clear arrays
    node.children_nodes = None
    node.children_priors = None
    node.children_visits = None
    node.children_values = None
    node.children_exists = None

# Visit count proportion threshold: if RED's turn-1 move gets more than this % of MCTS visits, Blue is forced to swap.
SWAP_PI_THRESHOLD = 0.10


def _decide_swap_turn2(game, swap_idx, mcts_root):
    """Decide whether Blue should invoke the swap rule on turn 2.

    Strategy
    --------
    We check the raw MCTS visit proportion that RED's opener received 
    during turn 1. That value lives in ``game['red_opener_pi']``.

    If RED's opener was highly visited (> SWAP_PI_THRESHOLD), Blue swaps.
    Otherwise Blue plays normally (sample from pi).

    Returns
    -------
    int  — the chosen action index (swap_idx or a normal board move)
    """
    red_opener_pi = game.get('red_opener_pi', None)

    if red_opener_pi is not None and red_opener_pi > SWAP_PI_THRESHOLD:
        logger.debug(
            f"[Swap] RED opener pi={red_opener_pi:.3f} > {SWAP_PI_THRESHOLD} → forcing SWAP"
        )
        return swap_idx

    # RED's opener is not dominant — let Blue decide via its own MCTS policy
    # (pi has already been computed for this turn; caller will fall through to multinomial)
    return None  # sentinel: caller uses multinomial


def self_play(model, buffer, num_games=GAMES_PER_EPOCH, mcts_simulations=CURRENT_SIMS,
              opponent_game_every=OPPONENT_GAME_EVERY, iteration=0):
    """
    Run a batch of self-play games, with diversity injection:
    - Every `opponent_game_every` games (0-indexed), play against a randomly
      chosen local agent instead of self-play.  These games are sequential
      (cannot be batched) so they run before the batched loop.
    - The remaining games run as batched MCTS self-play.

    Args:
        opponent_game_every: one out of this many games uses a local agent
                             opponent (e.g. 10 → game indices 0, 10, 20 …).
                             Set to 0 to disable.
    """
    model.eval()

    # --- 1. Identify which game slots are opponent games ---
    opponent_indices = set()
    if opponent_game_every and opponent_game_every > 0:
        opponent_indices = {i for i in range(num_games) if i % opponent_game_every == 0}

    num_opponent_games = len(opponent_indices)
    num_self_play_games = num_games - num_opponent_games

    finished_winners = []

    # --- 2. Sequential opponent-diversity games ---
    if num_opponent_games > 0:
        logger.info(f"Running {num_opponent_games} opponent-diversity game(s)...")
        for i in range(num_opponent_games):
            if _check_shutdown():
                logger.info("Shutdown requested during opponent games. Saving partial results...")
                break
            winner = play_vs_agent(model, buffer, mcts_simulations=mcts_simulations)
            finished_winners.append(winner)
            print(f"Opponent game {i+1}/{num_opponent_games} done (winner={winner})     ", end='\r')

    # --- 3. Batched self-play games ---
    if num_self_play_games > 0:
        # Every 10th self-play game (0-indexed within self-play slots) uses
        # full_expansion=True: the MCTS expands ALL valid moves at every node
        # instead of capping at top-16+4.  This acts as an "exploration reset"
        # that prevents the buffer from becoming an echo chamber.
        active_games = []
        for sp_idx in range(num_self_play_games):
            if iteration < 200:
                is_exploratory = True
            else:
                is_exploratory = (sp_idx % EXPLORATORY_EVERY == 0)
            active_games.append({
                'board': Board(BOARD_SIZE),
                'colour': Colour.RED,
                'turn': 1,
                'history': [],
                'root': Node(0),
                'is_exploratory': is_exploratory,
            })

        num_exploratory = sum(1 for g in active_games if g['is_exploratory'])
        logger.info(
            f"Self-play batch: {num_self_play_games} games "
            f"({num_exploratory} exploratory / {num_self_play_games - num_exploratory} normal)"
        )

        # Two MCTS instances — one per expansion mode.  Both are stateless
        # (no tree reuse across games) so sharing is safe.
        mcts_normal = BatchedMCTS(
            model, num_simulations=mcts_simulations, c_puct=C_PUCT,
            temperature=MCTS_TEMPERATURE, max_expansion_width=MAX_EXPANSION_WIDTH
        )
        mcts_exploratory = BatchedMCTS(
            model, num_simulations=mcts_simulations, c_puct=C_PUCT,
            temperature=MCTS_TEMPERATURE, max_expansion_width=None
        )

        while active_games:
            # Check for shutdown before each batch
            if _check_shutdown():
                logger.info("Shutdown requested during self-play. Discarding partial games and stopping...")
                # Do NOT save partial games: winner=None would label all moves as
                # losses (z=-1), polluting the replay buffer with false negatives.
                for game in active_games:
                    recursive_free(game['root'])
                break

            # Split by expansion mode and search each sub-batch separately
            normal_games      = [g for g in active_games if not g['is_exploratory']]
            exploratory_games = [g for g in active_games if     g['is_exploratory']]

            # Maps game-object → its pi tensor so we can reunify below
            pi_map = {}
            if normal_games:
                pis = mcts_normal.search(normal_games)
                for g, pi in zip(normal_games, pis):
                    pi_map[id(g)] = pi
            if exploratory_games:
                pis = mcts_exploratory.search(exploratory_games)
                for g, pi in zip(exploratory_games, pis):
                    pi_map[id(g)] = pi

            batch_pis = [pi_map[id(g)] for g in active_games]

            next_active = []
            for idx, game in enumerate(active_games):
                raw_pi = batch_pis[idx]
                root = game['root']

                valid_moves = get_valid_moves(game['board'], game['turn'])

                # --- Temperature schedule ---
                # BatchedMCTS returns raw visit proportions (temperature=1.0 inside MCTS).
                # We apply the per-turn sharpening here for action selection, but we MUST 
                # keep the raw_pi for the replay buffer to preserve MCTS exploration value.
                turn = game['turn']
                temp = (TEMP_HIGH if turn < TEMP_HIGH_TURNS
                        else TEMP_MID if turn < TEMP_MID_TURNS
                        else TEMP_LOW)
                
                if temp != 1.0:
                    sample_pi = raw_pi ** (1.0 / temp)
                    sample_pi = sample_pi / sample_pi.sum()
                else:
                    sample_pi = raw_pi

                swap_idx = BOARD_SIZE * BOARD_SIZE
                # ----------------------------------------------------------------
                # Smart swap decision (turn 2 only)
                # ----------------------------------------------------------------
                if turn == 2 and swap_idx in valid_moves:
                    swap_action = _decide_swap_turn2(game, swap_idx, root)
                    action = swap_action if swap_action is not None else torch.multinomial(sample_pi, 1).item()
                else:
                    action = torch.multinomial(sample_pi, 1).item()

                # After turn 1 (RED's opener), record the pi of the chosen
                # move so turn-2 Blue can decide whether to invoke the swap rule.
                if turn == 1:
                    game['red_opener_pi'] = float(raw_pi[action])

                device = next(model.parameters()).device
                
                # Note: Neural Network only needs to know turn 2 for the swap-rule signal
                state_tensor = encode_state(game['board'], game['colour'], device,
                                            turn=turn if turn == 2 else None).cpu()

                # CRITICAL: We append the RAW unsharpened MCTS probabilities to the history!
                game['history'].append((state_tensor, raw_pi.numpy(), game['colour'], action, valid_moves))

                new_board, next_col, next_turn, is_terminal, winner = apply_move(
                    game['board'], game['colour'], game['turn'], action
                )

                if is_terminal:
                    finished_winners.append(winner)
                    buffer.save_game(game['history'], winner)

                    recursive_free(game['root'])
                    game['root'] = None
                    game['history'].clear()

                    continue
                else:
                    game['board'] = new_board
                    game['colour'] = next_col
                    game['turn'] = next_turn
                    # Keep the tree! Step into the child node for the chosen action.
                    if game['root'].children_exists[action] and game['root'].children_nodes[action] is not None:
                        old_root = game['root']
                        new_root = old_root.children_nodes[action]
                        game['root'] = new_root
                        new_root.parent = None
                        old_root.children_nodes[action] = None
                        # Break references so GC can clean up old tree
                        recursive_free(old_root)
                        del old_root
                    else:
                        game['root'] = Node(0)  # Only fallback if something weird happens
                    next_active.append(game)

            active_games = next_active
            print(f"Self-play active games remaining: {len(active_games)}      ", end='\r')

        del active_games

    red_wins  = sum(1 for w in finished_winners if w == Colour.RED)
    blue_wins = sum(1 for w in finished_winners if w == Colour.BLUE)
    draws     = sum(1 for w in finished_winners if w is None)
    logger.info(
        f"Self-Play Batch Completed. RED Wins: {red_wins}, BLUE Wins: {blue_wins}, "
        f"Draws: {draws}  (incl. {num_opponent_games} opponent-diversity game(s))"
    )

    del finished_winners
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return buffer

def evaluate_batched(temp_model, best_model, num_games=NUM_GAMES_EVAL, eval_sims=EVAL_SIMS):
    temp_model.eval()
    best_model.eval()
    
    # Initialize all games simultaneously
    games = []
    for i in range(num_games):
        games.append({
            'board': Board(BOARD_SIZE),
            'colour': Colour.RED,
            'turn': 1,
            'temp_is_red': (i < num_games // 2), # Balanced sides
            'root': Node(0),
            'is_terminal': False,
            'winner': None
        })

    active_games = games
    
    while active_games:
        # Check for shutdown before each evaluation batch
        if _check_shutdown():
            logger.info("Shutdown requested during evaluation. Returning current results...")
            break
            
        # 1. SPLIT THE WORKLOAD
        # Who is moving right now?
        temp_batch = [g for g in active_games if (g['colour'] == Colour.RED and g['temp_is_red']) 
                      or (g['colour'] == Colour.BLUE and not g['temp_is_red'])]
        best_batch = [g for g in active_games if g not in temp_batch]

        # 2. BATCHED SEARCH (No Noise, Low Temperature)
        # We reuse your BatchedMCTS class
        if temp_batch:
            mcts_temp = BatchedMCTS(temp_model, num_simulations=eval_sims, c_puct=C_PUCT, temperature=MCTS_TEMPERATURE_EVAL, add_noise=False, max_expansion_width=MAX_EXPANSION_WIDTH)
            batch_pis_temp = mcts_temp.search(temp_batch)
            for idx, g in enumerate(temp_batch):
                g['pi'] = batch_pis_temp[idx]

        if best_batch:
            mcts_best = BatchedMCTS(best_model, num_simulations=eval_sims, c_puct=C_PUCT, temperature=MCTS_TEMPERATURE_EVAL, add_noise=False, max_expansion_width=MAX_EXPANSION_WIDTH)
            batch_pis_best = mcts_best.search(best_batch)
            for idx, g in enumerate(best_batch):
                g['pi'] = batch_pis_best[idx]

        # 3. APPLY MOVES
        next_active = []
        for g in active_games:
            # BatchedMCTS already applies MCTS_TEMPERATURE_EVAL when building the
            # visit-count distribution, so g['pi'] is already appropriately sharpened.
            # Sample directly — no second sharpening (that would be double-application).
            action = torch.multinomial(g['pi'], 1).item()
            new_board, next_col, next_turn, is_term, win = apply_move(
                g['board'], g['colour'], g['turn'], action
            )
            
            g['board'], g['colour'], g['turn'], g['is_terminal'], g['winner'] = \
                new_board, next_col, next_turn, is_term, win
            
            if not is_term:
                # Keep the tree! Step into the child node for the chosen action.
                if g['root'].children_exists[action] and g['root'].children_nodes[action] is not None:
                    old_root = g['root']
                    new_root = old_root.children_nodes[action]
                    g['root'] = new_root
                    # --- THE FIX: Detach the new root here too ---
                    old_root.children_nodes[action] = None
                    
                    # Break references so GC can clean up old tree
                    recursive_free(old_root)
                    del old_root
                else:
                    g['root'] = Node(0)  # Only fallback if something weird happens
                
                next_active.append(g)
        
        active_games = next_active

    # 4. CALCULATE WIN RATE
    # Exclude draws: a draw is neither a win nor a loss for either model, so
    # counting it as a loss for the temp model biases the threshold downward.
    temp_wins = 0
    best_wins = 0
    decided = 0
    draws = 0
    for g in games:
        if g['winner'] is None:
            draws += 1
            continue  # draw — exclude from denominator
        decided += 1
        if (g['winner'] == Colour.RED and g['temp_is_red']) or \
           (g['winner'] == Colour.BLUE and not g['temp_is_red']):
            temp_wins += 1
        else:
            best_wins += 1

    effective_games = decided if decided > 0 else num_games
    win_rate = temp_wins / effective_games

    eval_summary = (
        f"Eval | Temp win rate: {win_rate*100:.1f}% "
        f"| Temp: {temp_wins}W  Best: {best_wins}W  Draws: {draws} "
        f"| {decided}/{num_games} decided"
    )
    # Route to both loggers: main log gets a brief note, eval log gets the full detail
    logger.info(eval_summary)
    eval_logger.info(eval_summary)

    del games
    del active_games
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return win_rate >= EVAL_WIN_RATE_THRESHOLD


def _save_and_exit(best_model, trainer, buffer, start_iteration, current_epoch):
    """Save all state and exit gracefully."""
    try:
        # Save the replay buffer
        buffer.save()
        logger.info("Replay buffer saved.")
        
        # Save model checkpoint
        trainer.save_checkpoint(current_epoch + 1)
        torch.save({
            'iteration': current_epoch + 1,
            'board_size': BOARD_SIZE,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        logger.info("Model checkpoint saved.")
        
        logger.info(f"Graceful shutdown complete. Reached epoch {current_epoch + 1}.")
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")
    
    sys.exit(0)

def run_training():
    """Main training loop.  Call configure() beforehand to override defaults."""
    global EVALS_PASSED, CURRENT_SIMS
    _setup_logging()

    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize best model
    best_model = HexPVNet(board_size=BOARD_SIZE).to(device)

    trainer = HexTrainer(best_model)
    iteration = trainer.load_checkpoint()

    # Initialize ReplayBuffer globally so it persists across epochs
    buffer = ReplayBuffer()
    buffer.load()

    # Initialize temporary model for training (should be separate instance)
    temp_model = HexPVNet(board_size=BOARD_SIZE).to(device)
    temp_model.load_state_dict(best_model.state_dict())
    temp_trainer = HexTrainer(temp_model)

    for epoch in range(iteration, iteration + EPOCHS):
        logger.info(f"--- Epoch {epoch+1} ---")

        # 1. Self Play
        logger.info(f"Starting Self-Play (sims={CURRENT_SIMS})...")
        self_play(best_model, buffer, num_games=GAMES_PER_EPOCH, mcts_simulations=CURRENT_SIMS, iteration=epoch)

        # Save buffer periodically
        buffer.save()

        # 2. Train Temp Model
        logger.info("Starting Training...")

        # Train for some iterations on the buffer
        # We need enough tuples to train
        valid_buffer_size = len(buffer)
        if valid_buffer_size >= BATCH_SIZE:
            total_loss = total_p = total_v = 0.0

            for b in range(TRAINING_STEPS):
                states, pis, zs, actions, valid_moves = buffer.sample(BATCH_SIZE)
                loss, p_loss, v_loss = temp_trainer.train_step(states, pis, zs, valid_moves)
                total_loss += loss
                total_p += p_loss
                total_v += v_loss

            logger.info(f"Training Loss: {total_loss/TRAINING_STEPS:.4f} (P: {total_p/TRAINING_STEPS:.4f}, "
                        f"V: {total_v/TRAINING_STEPS:.4f}) | Valid Buffer: {valid_buffer_size}")

        # 3. Evaluate (only every EVAL_EVERY epochs)
        if (epoch - iteration) % EVAL_EVERY == 0:
            eval_header = f"=== Evaluation | Epoch {epoch+1} | sims={EVAL_SIMS} ==="
            logger.info(f"Evaluating (sims={EVAL_SIMS})...")
            eval_logger.info(eval_header)
            is_better = evaluate_batched(temp_model, best_model, NUM_GAMES_EVAL, eval_sims=EVAL_SIMS)

            if is_better:
                outcome = "ACCEPTED — Temp model promoted to best_model."
                logger.info("Temp model is better! Saving as new best_model.")
                eval_logger.info(outcome)

                best_model.load_state_dict(temp_model.state_dict())
                
                EVALS_PASSED += 1
                # Trigger Simulation Decay every 2 successful evaluations
                if EVALS_PASSED % 2 == 0 and CURRENT_SIMS > MIN_SIMS:
                    CURRENT_SIMS -= SIMS_DECAY_AMOUNT
                    CURRENT_SIMS = max(CURRENT_SIMS, MIN_SIMS) # Don't drop below floor
                    level_up_msg = f"-> NETWORK LEVELED UP. Decreasing Self-Play Sims to {CURRENT_SIMS}"
                    logger.info(level_up_msg)
                    eval_logger.info(level_up_msg)

                trainer.save_checkpoint(epoch+1)
                torch.save({
                    'iteration': epoch+1,
                    'board_size': BOARD_SIZE,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

            else:
                outcome = "REJECTED — Temp model reverted to best_model."
                logger.info("Temp model rejected. Reverting temp_model to best_model.")
                eval_logger.info(outcome)
                temp_model.load_state_dict(best_model.state_dict())

        print("\n")

        # Check for graceful shutdown after each epoch
        if _check_shutdown():
            logger.info("Shutdown requested. Saving state and exiting...")
            _save_and_exit(best_model, trainer, buffer, iteration, epoch)

    logger.info("Training completed normally.")


if __name__ == "__main__":
    run_training()



