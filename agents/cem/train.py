import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import gc
from collections import deque

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.cem.agent1 import Hex3HNN
from agents.cem.mcts import MCTS, BatchedMCTS, Node, encode_state, get_valid_moves, apply_move

import logging
#TODO - Add graceful termination
#TODO - Add notifications on epoch completion (discord, desktop etc.)
#TODO: replay buffer wastes RAM by storing PyTorch tensors. We should convert to numpy before saving and back to tensor when sampling.
#TODO: BatchedMCTS is currently a bottleneck due to PyTorch overhead. implement iterative backprop. look to replace "search_paths.append(path)"
#TODO: "g['root'] = Node(0)" is a fallback that should never be hit. If it is, we leak the old subtree entirely. can consider g['root'] = None
#TODO: overall look into memory leaks. might look into how alpha-zero implementations handle this


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# --- Replay Buffer ---
REPLAY_BUFFER_CAPACITY = 100000  # Max games to store in buffer

# --- Training (Optimizer) ---
LEARNING_RATE = 0.001           # Adam optimizer learning rate
WEIGHT_DECAY = 1e-4             # L2 regularization strength

# --- Training Loop ---
EPOCHS = 300                   # Total training epochs
GAMES_PER_EPOCH = 12            # Self-play games per epoch
BATCH_SIZE = 256                # Training batch size
TRAINING_STEPS = 200            # Gradient updates per epoch
EVAL_EVERY = 10                 # Evaluate every N epochs
NUM_GAMES_EVAL = 24             # Games for evaluation

# --- MCTS Simulations ---
SELF_PLAY_SIMS = 50             # MCTS simulations per move during self-play
EVAL_SIMS = 150                 # MCTS simulations per move during evaluation
OPPONENT_GAME_SIMS = 100        # MCTS simulations per move vs local agents

# --- MCTS Settings ---
MCTS_TEMPERATURE = 1.0          # Exploration temperature for self-play
MCTS_TEMPERATURE_EVAL = 0.1     # Exploration temperature for evaluation
ADD_NOISE = True                # Add Dirichlet noise to root (self-play only)

# --- Temperature Schedule (for move selection) ---
TEMP_HIGH_TURNS = 10            # Use high temp for first N turns
TEMP_MID_TURNS = 20             # Use mid temp for next N turns
TEMP_HIGH = 1.0                # High temperature value
TEMP_MID = 0.5                  # Medium temperature value
TEMP_LOW = 0.1                  # Low temperature value (greedy)

# --- Opponent Diversity ---
OPPONENT_GAME_EVERY = 10        # Play vs local agent every N games (0 to disable)

# --- Evaluation ---
EVAL_WIN_RATE_THRESHOLD = 0.50  # Win rate needed to replace best model (50%)

# --- Board Size (fixed for now) ---
BOARD_SIZE = 11


os.makedirs("agents/cem", exist_ok=True)
logger = logging.getLogger("HexTraining")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
formatter.converter = time.localtime

fh = logging.FileHandler("agents/cem/training.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

import pickle # Ensure this is imported at the top

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
        
    def save_game(self, game_history, winner_colour):
        for state, pi, target_q, current_colour, action, valid_moves in game_history:
            z = 1.0 if current_colour == winner_colour else -1.0
            
            # 1. Convert state to numpy to avoid massive PyTorch storage overhead
            state_np = state.numpy() if isinstance(state, torch.Tensor) else state
            
            self.buffer.append((state_np, pi, target_q, z, action, valid_moves))
            
            # Data Augmentation: Hex 180-degree rotation using NumPy
            rotated_state_np = np.rot90(state_np, 2, axes=(2, 3)).copy()
            
            pi_board = pi[:-1].reshape(11, 11)
            rotated_pi_board = np.rot90(pi_board, 2)
            rotated_pi = np.append(rotated_pi_board.flatten(), pi[-1]) 
            
            q_board = target_q[:-1].reshape(11, 11)
            rotated_q_board = np.rot90(q_board, 2)
            rotated_q = np.append(rotated_q_board.flatten(), target_q[-1])
            
            if action != 121:
                x, y = divmod(action, 11)
                rx, ry = 10 - x, 10 - y
                rotated_action = rx * 11 + ry
            else:
                rotated_action = 121
                
            rotated_valid_moves = []
            for vm in valid_moves:
                if vm != 121:
                    vx, vy = divmod(vm, 11)
                    rx, ry = 10 - vx, 10 - vy
                    rotated_valid_moves.append(rx * 11 + ry)
                else:
                    rotated_valid_moves.append(121)
                    
            self.buffer.append((rotated_state_np, rotated_pi, rotated_q, z, rotated_action, rotated_valid_moves))
            
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty.")

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, pis, qs, zs, actions, valid_moves_list = zip(*batch)
        
        states_tensor = torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32)
        pis_tensor = torch.tensor(np.array(pis), dtype=torch.float32)
        qs_tensor = torch.tensor(np.array(qs), dtype=torch.float32)
        zs_tensor = torch.tensor(zs, dtype=torch.float32)
        
        return states_tensor, pis_tensor, qs_tensor, zs_tensor, actions, valid_moves_list
        
    def __len__(self):
        return len(self.buffer)
        
    def save(self, filename):
        # 3. Use standard pickle for lists of standard python/numpy objects
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        
    def load(self, filename):
        if os.path.isfile(filename):
            try:
                # Try loading the new pickle format first
                with open(filename, 'rb') as f:
                    loaded = pickle.load(f)
            except Exception:
                # Fallback: Load your old PyTorch format buffer
                loaded = torch.load(filename, map_location="cpu", weights_only=False)
            
            # 4. Clean up any lingering PyTorch tensors from an older save file
            cleaned_loaded = []
            for item in loaded:
                state, pi, target_q, z, action, valid_moves = item
                state_np = state.numpy() if isinstance(state, torch.Tensor) else state
                cleaned_loaded.append((state_np, pi, target_q, z, action, valid_moves))
                
            self.buffer = deque(cleaned_loaded, maxlen=self.buffer.maxlen)
            print(f"Loaded {len(self.buffer)} games from {filename} and normalized states.")


class HexTrainer:
    def __init__(self, model, lr=LEARNING_RATE):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.history = {'loss': [], 'v_loss': [], 'p_loss': [], 'q_loss': []}

    def train_step(self, states, target_pis, target_qs, target_vs, actions, valid_moves_list):
        self.model.train()
        device = next(self.model.parameters()).device
        states = states.to(device)
        target_pis = target_pis.to(device)
        target_qs = target_qs.to(device)
        target_vs = target_vs.to(device)

        self.optimizer.zero_grad()

        # Forward pass
        p_logits, v, q = self.model(states)

        # 1. Masking Invalid Moves in Policy
        # Create a boolean mask of the same shape as p_logits (Batch, 122)
        mask = torch.ones_like(p_logits, dtype=torch.bool)
        for i, valid in enumerate(valid_moves_list):
            mask[i, valid] = False # False means it IS a valid move
            
        # Overwrite illegal logits with a massive negative number
        p_logits = p_logits.masked_fill(mask, -1e9)

        # Now the softmax will perfectly ignore illegal moves
        p_loss = -torch.mean(torch.sum(target_pis * F.log_softmax(p_logits, dim=1), dim=1))

        # 2. State-Value Loss (MSE)
        v_loss = F.mse_loss(v.view(-1), target_vs)

        # 3. Action-Value Loss (MSE)
        # We also want to mask invalid moves for the Q-loss so we don't train on garbage
        q = q.masked_fill(mask, 0.0)
        target_qs = target_qs.masked_fill(mask, 0.0)
        q_loss = F.mse_loss(q, target_qs)

        total_loss = p_loss + v_loss + q_loss
        total_loss.backward()
        self.optimizer.step()

        self.history['loss'].append(total_loss.item())
        self.history['p_loss'].append(p_loss.item())
        self.history['v_loss'].append(v_loss.item())
        self.history['q_loss'].append(q_loss.item())

        return total_loss.item(), p_loss.item(), v_loss.item(), q_loss.item()

    def save_checkpoint(self, iteration, path="agents/cem/"):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path, f"checkpoint_{iteration}.pt"))
        
    def load_checkpoint(self, filename):
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
    board = Board(11)

    # Randomly decide which side the model plays
    model_colour = random.choice([Colour.RED, Colour.BLUE])
    opp_colour   = Colour.opposite(model_colour)

    # Instantiate the random opponent agent
    opp_agent = load_random_local_agent(opp_colour)
    logger.debug(f"Opponent-diversity game: model={model_colour}, opp={type(opp_agent).__name__}")

    mcts = MCTS(model, num_simulations=mcts_simulations, temperature=MCTS_TEMPERATURE)

    history      = []          # (state_tensor, pi, target_q, colour, action, valid_moves)
    current_colour = Colour.RED
    turn           = 1
    last_move: Move | None = None
    winner         = None

    while True:
        if current_colour == model_colour:
            # ---- Model's turn: use MCTS ----
            pi_tensor = mcts.search(board, current_colour, turn)
            root      = mcts  # MCTS.search returns action_probs only; we need Q-values from the root

            # Rebuild root data for Q-target (MCTS.search builds a fresh tree internally)
            # We call search again on the same board for Q-values — but that's expensive.
            # Instead, encode once for Q extraction.
            device = next(model.parameters()).device
            with torch.no_grad():
                state_t = encode_state(board, current_colour, device)
                _, _, q_pred = model(state_t)
            target_q = q_pred[0].cpu().numpy().astype(np.float32)  # shape (122,)

            valid_moves = get_valid_moves(board, turn)

            temp = TEMP_HIGH if turn < TEMP_HIGH_TURNS else TEMP_MID if turn < TEMP_MID_TURNS else TEMP_LOW
            if temp == 0.1:
                action = int(torch.argmax(pi_tensor).item())
            else:
                action = int(torch.multinomial(pi_tensor, 1).item())

            state_cpu = encode_state(board, current_colour, device).cpu()
            history.append((state_cpu, pi_tensor.numpy(), target_q, current_colour, action, valid_moves))

            # Apply the model's move
            move_obj = Move(-1, -1) if action == 121 else Move(*divmod(action, 11))
            new_board, next_col, next_turn, is_terminal, winner = apply_move(
                board, current_colour, turn, action
            )
            last_move = move_obj
        else:
            # ---- Opponent agent's turn ----
            move_obj = opp_agent.make_move(turn, board, last_move)

            if move_obj.x == -1 and move_obj.y == -1:
                # Swap move — action index 121
                action = 121
            else:
                action = move_obj.x * 11 + move_obj.y

            # Validate: if the move lands on an occupied cell, skip (play first valid)
            if action != 121 and action not in get_valid_moves(board, turn):
                valid = get_valid_moves(board, turn)
                if not valid:
                    break  # Should not happen, but guard
                action = valid[0]
                move_obj = Move(*divmod(action, 11))

            new_board, next_col, next_turn, is_terminal, winner = apply_move(
                board, current_colour, turn, action
            )
            last_move = move_obj

        board          = new_board
        current_colour = next_col
        turn           = next_turn

        if is_terminal:
            break

        # Safety valve: game should never last this long on 11x11
        if turn > 125:
            logger.warning("play_vs_agent: hit turn limit 125 without terminal — forcing stop")
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
    node.children_q_priors = None
    node.children_exists = None

def self_play(model, buffer, num_games=GAMES_PER_EPOCH, mcts_simulations=SELF_PLAY_SIMS,
              opponent_game_every=OPPONENT_GAME_EVERY):
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
            winner = play_vs_agent(model, buffer, mcts_simulations=mcts_simulations)
            finished_winners.append(winner)
            print(f"Opponent game {i+1}/{num_opponent_games} done (winner={winner})     ", end='\r')

    # --- 3. Batched self-play games ---
    if num_self_play_games > 0:
        active_games = []
        for _ in range(num_self_play_games):
            active_games.append({
                'board': Board(11),
                'colour': Colour.RED,
                'turn': 1,
                'history': [],
                'root': Node(0)
            })

        mcts = BatchedMCTS(model, num_simulations=mcts_simulations, temperature=MCTS_TEMPERATURE)

        while active_games:
            batch_pis = mcts.search(active_games)

            next_active = []
            for idx, game in enumerate(active_games):
                pi = batch_pis[idx]
                root = game['root']

                # Extract true MCTS Q-values
                target_q = np.zeros(122, dtype=np.float32)
                visits = root.children_visits
                visited = visits > 0
                target_q[visited] = root.children_values[visited] / visits[visited]
                target_q[~visited] = root.children_q_priors[~visited]
                target_q[~root.children_exists] = 0.0

                valid_moves = get_valid_moves(game['board'], game['turn'])

                # High temp for first N moves, then greedy to finish strong
                temp = TEMP_HIGH if game['turn'] < TEMP_HIGH_TURNS else TEMP_MID if game['turn'] < TEMP_MID_TURNS else TEMP_LOW

                if temp == 0.1:
                    action = torch.argmax(pi).item()
                else:
                    action = torch.multinomial(pi, 1).item()

                device = next(model.parameters()).device
                state_tensor = encode_state(game['board'], game['colour'], device).cpu()

                # NOTE: We now pass target_q to the history instead of calculating it later
                game['history'].append((state_tensor, pi.numpy(), target_q, game['colour'], action, valid_moves))

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
            'board': Board(11),
            'colour': Colour.RED,
            'turn': 1,
            'temp_is_red': (i < num_games // 2), # Balanced sides
            'root': Node(0),
            'is_terminal': False,
            'winner': None
        })

    active_games = games
    
    while active_games:
        # 1. SPLIT THE WORKLOAD
        # Who is moving right now?
        temp_batch = [g for g in active_games if (g['colour'] == Colour.RED and g['temp_is_red']) 
                      or (g['colour'] == Colour.BLUE and not g['temp_is_red'])]
        best_batch = [g for g in active_games if g not in temp_batch]

        # 2. BATCHED SEARCH (No Noise, Low Temperature)
        # We reuse your BatchedMCTS class
        if temp_batch:
            mcts_temp = BatchedMCTS(temp_model, num_simulations=eval_sims, temperature=MCTS_TEMPERATURE_EVAL, add_noise=False)
            batch_pis_temp = mcts_temp.search(temp_batch)
            for idx, g in enumerate(temp_batch):
                g['pi'] = batch_pis_temp[idx]

        if best_batch:
            mcts_best = BatchedMCTS(best_model, num_simulations=eval_sims, temperature=MCTS_TEMPERATURE_EVAL, add_noise=False)
            batch_pis_best = mcts_best.search(best_batch)
            for idx, g in enumerate(best_batch):
                g['pi'] = batch_pis_best[idx]

        # 3. APPLY MOVES
        next_active = []
        for g in active_games:
            action = torch.argmax(g['pi']).item() # Always greedy in eval
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
    temp_wins = 0
    for g in games:
        if (g['winner'] == Colour.RED and g['temp_is_red']) or \
           (g['winner'] == Colour.BLUE and not g['temp_is_red']):
            temp_wins += 1

    win_rate = temp_wins / num_games
    logger.info(f"Evaluator: Temp model win rate: {win_rate*100:.1f}%")
    del games
    del active_games
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return win_rate >= EVAL_WIN_RATE_THRESHOLD

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize best model
    best_model = Hex3HNN(board_size=11).to(device)
    
    trainer = HexTrainer(best_model)
    iteration = trainer.load_checkpoint("agents/cem/best_model.pt")
    
    # Initialize ReplayBuffer globally so it persists across epochs
    buffer = ReplayBuffer()
    buffer.load("agents/cem/buffer.pt")

    # Initialize temporary model for training (should be separate instance)
    temp_model = Hex3HNN(board_size=BOARD_SIZE).to(device)
    temp_model.load_state_dict(best_model.state_dict())
    temp_trainer = HexTrainer(temp_model)
    
    for epoch in range(iteration, iteration + EPOCHS):
        logger.info(f"--- Epoch {epoch+1} ---")
        
        # 1. Self Play
        logger.info(f"Starting Self-Play (sims={SELF_PLAY_SIMS})...")
        self_play(best_model, buffer, num_games=GAMES_PER_EPOCH, mcts_simulations=SELF_PLAY_SIMS)
        
        # Save buffer periodically
        buffer.save("agents/cem/buffer.pt")
        
        # 2. Train Temp Model
        logger.info("Starting Training...")
        
        # Train for some iterations on the buffer
        # We need enough 6-element tuples (with target_q) to train
        valid_buffer_size = sum(1 for b in buffer.buffer if len(b) == 6)
        if valid_buffer_size >= BATCH_SIZE:
            random.shuffle(buffer.buffer)
            total_loss = total_p = total_v = total_q = 0.0
            
            for b in range(TRAINING_STEPS):
                states, pis, qs, zs, actions, valid_moves = buffer.sample(BATCH_SIZE)
                loss, p_loss, v_loss, q_loss = temp_trainer.train_step(states, pis, qs, zs, actions, valid_moves)
                total_loss += loss
                total_p += p_loss
                total_v += v_loss
                total_q += q_loss
                
            logger.info(f"Training Loss: {total_loss/TRAINING_STEPS:.4f} (P: {total_p/TRAINING_STEPS:.4f}, "
                        f"V: {total_v/TRAINING_STEPS:.4f}, Q: {total_q/TRAINING_STEPS:.4f}) | Valid Buffer: {valid_buffer_size}")
        
        # 3. Evaluate (only every EVAL_EVERY epochs)
        if (epoch - iteration) % EVAL_EVERY == 0:
            logger.info(f"Evaluating (sims={EVAL_SIMS})...")
            is_better = evaluate_batched(temp_model, best_model, NUM_GAMES_EVAL, eval_sims=EVAL_SIMS)
            
            if is_better:
                logger.info("Temp model is better! Saving as new best_model.")

                best_model.load_state_dict(temp_model.state_dict())

                trainer.save_checkpoint(epoch+1, path="agents/cem/")
                torch.save({
                    'iteration': epoch+1,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                }, "agents/cem/best_model.pt")

            else:
                logger.info("Temp model rejected. Reverting temp_model to best_model.")
                temp_model.load_state_dict(best_model.state_dict())

        print("\n")
