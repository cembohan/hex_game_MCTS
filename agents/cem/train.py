import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import deque

from src.Board import Board
from src.Colour import Colour

from agents.cem.agent1 import Hex3HNN
from agents.cem.mcts import MCTS, BatchedMCTS, Node, encode_state, get_valid_moves, apply_move

import logging

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

class ReplayBuffer:
    def __init__(self, capacity=100000): # Increased capacity
        self.buffer = deque(maxlen=capacity)
        
    def save_game(self, game_history, winner_colour):
        # game_history is a list of (state_tensor, pi, target_q, current_colour, action, valid_moves)
        for state, pi, target_q, current_colour, action, valid_moves in game_history:
            # Value is +1 if the player who made the move won, else -1
            z = 1.0 if current_colour == winner_colour else -1.0
            self.buffer.append((state, pi, target_q, z, action, valid_moves))
            
            # Data Augmentation: Hex 180-degree rotation
            # For player-relative encoding, we just rotate the spatial dimensions
            rotated_state = torch.rot90(state, 2, [2, 3])
            
            # Rotate the policy (excluding the swap move at the last index)
            pi_board = pi[:-1].reshape(11, 11)
            rotated_pi_board = np.rot90(pi_board, 2)
            rotated_pi = np.append(rotated_pi_board.flatten(), pi[-1]) 
            
            # Rotate target_q exactly like policy
            q_board = target_q[:-1].reshape(11, 11)
            rotated_q_board = np.rot90(q_board, 2)
            rotated_q = np.append(rotated_q_board.flatten(), target_q[-1])
            
            # Action adjustment for rotated board
            if action != 121:
                x, y = divmod(action, 11)
                rx, ry = 10 - x, 10 - y
                rotated_action = rx * 11 + ry
            else:
                rotated_action = 121
                
            # valid moves adjustment
            rotated_valid_moves = []
            for vm in valid_moves:
                if vm != 121:
                    vx, vy = divmod(vm, 11)
                    rx, ry = 10 - vx, 10 - vy
                    rotated_valid_moves.append(rx * 11 + ry)
                else:
                    rotated_valid_moves.append(121)
                    
            self.buffer.append((rotated_state, rotated_pi, rotated_q, z, rotated_action, rotated_valid_moves))
            
    def sample(self, batch_size):
        # Filter out old 5-element buffer entries if mixing buffers
        valid_buffer = [b for b in self.buffer if len(b) == 6]
        if not valid_buffer:
            raise ValueError("Buffer has no 6-element tuples (with target_q). Please delete buffer.pt and restart.")
        batch = random.sample(valid_buffer, min(batch_size, len(valid_buffer)))
        
        states, pis, qs, zs, actions, valid_moves_list = zip(*batch)
        
        # Concat states
        states_tensor = torch.cat([s.cpu() for s in states], dim=0) # (Batch, 3, 11, 11)
        pis_tensor = torch.tensor(np.array(pis), dtype=torch.float32) # (Batch, 122)
        qs_tensor = torch.tensor(np.array(qs), dtype=torch.float32) # (Batch, 122)
        zs_tensor = torch.tensor(zs, dtype=torch.float32) # (Batch,)
        
        return states_tensor, pis_tensor, qs_tensor, zs_tensor, actions, valid_moves_list
        
    def __len__(self):
        return len(self.buffer)
        
    def save(self, filename):
        torch.save(list(self.buffer), filename)
        
    def load(self, filename):
        if os.path.isfile(filename):
            loaded = torch.load(filename, map_location="cpu", weights_only=False)
            self.buffer = deque(loaded, maxlen=self.buffer.maxlen)
            print(f"Loaded {len(self.buffer)} games from {filename}")


class HexTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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


def self_play(model, buffer, num_games=10, mcts_simulations=100):
    model.eval()
    
    active_games = []
    finished_winners = []
    for _ in range(num_games):
        active_games.append({
            'board': Board(11),
            'colour': Colour.RED,
            'turn': 1,
            'history': [],
            'root': Node(0)
        })
        
    mcts = BatchedMCTS(model, num_simulations=mcts_simulations, temperature=1.0)
    
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
            
            # High temp for first 10 moves, then greedy to finish strong
            temp = 1.0 if game['turn'] < 10 else 0.5 if game['turn'] < 20 else 0.1
            
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
            else:
                game['board'] = new_board
                game['colour'] = next_col
                game['turn'] = next_turn
                # Keep the tree! Step into the child node for the chosen action.
                if game['root'].children_exists[action] and game['root'].children_nodes[action] is not None:
                    game['root'] = game['root'].children_nodes[action]
                    game['root'].parent = None # Cut off the old parent to save memory
                else:
                    game['root'] = Node(0) # Only fallback if something weird happens
                next_active.append(game)
                
        active_games = next_active
        print(f"Active games remaining: {len(active_games)}      ", end='\r')
        
    red_wins = sum(1 for w in finished_winners if w == Colour.RED)
    blue_wins = sum(1 for w in finished_winners if w == Colour.BLUE)
    draws = sum(1 for w in finished_winners if w is None)
    logger.info(f"Self-Play Batch Completed. RED Wins: {red_wins}, BLUE Wins: {blue_wins}, Draws: {draws}")
        
    return buffer

def evaluate_batched(temp_model, best_model, num_games=20, eval_sims=50):
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
            mcts_temp = BatchedMCTS(temp_model, num_simulations=eval_sims, temperature=0.1, add_noise=False)
            batch_pis_temp = mcts_temp.search(temp_batch)
            for idx, g in enumerate(temp_batch):
                g['pi'] = batch_pis_temp[idx]

        if best_batch:
            mcts_best = BatchedMCTS(best_model, num_simulations=eval_sims, temperature=0.1, add_noise=False)
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
                    g['root'] = g['root'].children_nodes[action]
                    g['root'].parent = None # Cut off the old parent to save memory
                else:
                    g['root'] = Node(0) # Only fallback if something weird happens
                
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
    return win_rate > 0.51

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
    
    # Training Loop
    EPOCHS = 100
    GAMES_PER_EPOCH = 12
    BATCH_SIZE = 256
    TRAINING_STEPS = 200
    EVAL_EVERY = 3  # Only evaluate every N epochs to save time
    
    for epoch in range(iteration, iteration + EPOCHS):
        logger.info(f"--- Epoch {epoch+1} ---")
        
        # MCTS Simulation logic
        if epoch < 10:
            sp_sims = 30
            eval_sims = 50
        elif epoch < 50:
            sp_sims = 30
            eval_sims = 30
        else:
            sp_sims = 100
            eval_sims = 200
            
        # 1. Self Play
        logger.info(f"Starting Self-Play (sims={sp_sims})...")
        self_play(best_model, buffer, num_games=GAMES_PER_EPOCH, mcts_simulations=sp_sims)
        
        # Save buffer periodically
        buffer.save("agents/cem/buffer.pt")
        
        # 2. Train Temp Model
        logger.info("Starting Training...")
        temp_model = Hex3HNN(board_size=11).to(device)
        temp_model.load_state_dict(best_model.state_dict())
        temp_trainer = HexTrainer(temp_model)
        
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
            logger.info(f"Evaluating (sims={eval_sims})...")
            is_better = evaluate_batched(temp_model, best_model, num_games=20, eval_sims=eval_sims)
            
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
                logger.info("Temp model rejected. Keeping previous best_model.")
        else:
            # Still copy temp model weights to best_model for next iteration's self-play
            best_model.load_state_dict(temp_model.state_dict())
            
        print("\n")
