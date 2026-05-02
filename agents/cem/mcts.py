import math
import copy
import torch
import torch.nn.functional as F
import numpy as np
import weakref

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Global board configuration
BOARD_SIZE = 11
NUM_ACTIONS = 122  # board_size * board_size + 1

def init_board_config(board_size):
    """Initialize MCTS with the target board size. Call this before creating MCTS instances."""
    global BOARD_SIZE, NUM_ACTIONS
    BOARD_SIZE = board_size
    NUM_ACTIONS = board_size * board_size + 1

class Node:
    def __init__(self, prior, parent=None, action_from_parent=None):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.q_prior = 0.0
        
        # FIX 1: Use weakref to break cyclic references
        self.parent = weakref.ref(parent) if parent is not None else None
        self.action_from_parent = action_from_parent
        
        self.is_expanded = False
        self.children_priors = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.children_visits = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.children_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.children_q_priors = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.children_exists = np.zeros(NUM_ACTIONS, dtype=bool)
        
        # FIX 2: Use standard lists instead of uninitialized numpy object arrays
        self.children_nodes = [None] * NUM_ACTIONS
        
    def value(self):
        if self.visit_count == 0:
            return self.q_prior
        return self.value_sum / self.visit_count

def get_valid_moves(board, turn):
    """Get valid moves using cached empty cells (O(1) instead of O(n²))."""
    valid = list(board.get_empty_cells())  # Fast copy of cached list
    # Turn 2 is the second move of the game, made by the second player
    if turn == 2:
        valid.append(board.size * board.size)  # Swap action
    return valid

_BOARD_POOL = []

def fast_clone_board(board):
    """Creates a fast shallow copy of the board for MCTS simulation without modifying src/Board.py"""
    if _BOARD_POOL:
        new_board = _BOARD_POOL.pop()
        new_board._winner = None
        new_board._winning_path.clear()
    else:
        from src.Board import Board
        new_board = Board(board.size)
        
    for i in range(board.size):
        for j in range(board.size):
            new_board.tiles[i][j].colour = board.tiles[i][j].colour
    # Copy empty cells cache
    new_board._empty_cells = list(board._empty_cells)
    # Copy numpy array cache
    new_board._color_array = board._color_array.copy()
    return new_board

def release_board(board):
    _BOARD_POOL.append(board)

def apply_move(board, current_colour, turn, action, inplace=False, check_win=True):
    if inplace:
        new_board = board
    else:
        new_board = fast_clone_board(board)
    board_size = board.size
    
    if action == board_size * board_size:
        # Swap: board remains the same.
        next_colour = Colour.BLUE
    else:
        x, y = divmod(action, board_size)
        new_board.set_tile_colour(x, y, current_colour)
        next_colour = Colour.opposite(current_colour)
        
    # Check win
    is_terminal = False
    winner = None
    if check_win and action != board_size * board_size:
        if new_board.has_ended(current_colour):
            is_terminal = True
            winner = current_colour
            
    return new_board, next_colour, turn + 1, is_terminal, winner

def encode_state(board, current_colour, device, out_tensor=None):
    """Encode board state using cached numpy array for O(1) vectorized encoding."""
    board_size = board.size
    if out_tensor is None:
        out_tensor = torch.zeros(1, 3, board_size, board_size, device=device)
    else:
        out_tensor.zero_()
    
    # Use cached numpy array - vectorized operations, no Python loops
    color_arr = board.get_color_array()
    
    if current_colour == Colour.RED:
        # RED's turn: channel 0 = RED stones, channel 1 = BLUE stones
        out_tensor[0, 0] = torch.from_numpy(color_arr == 1).float()
        out_tensor[0, 1] = torch.from_numpy(color_arr == 2).float()
        out_tensor[0, 2] = 1.0  # Current player indicator
    else:
        # BLUE's turn: channel 0 = BLUE stones (opponent), channel 1 = RED stones (current)
        out_tensor[0, 0] = torch.from_numpy(color_arr == 2).float()
        out_tensor[0, 1] = torch.from_numpy(color_arr == 1).float()
        # No channel 2 for BLUE (or could be 0)
        
    return out_tensor

class MCTS:
    def __init__(self, model, num_simulations=400, c_puct=2.0, temperature=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = next(self.model.parameters()).device
        # Pre-allocate buffer to avoid reallocation bottleneck
        self.state_buffer = torch.zeros(1, 3, self.model.board_size, self.model.board_size, device=self.device)
        
    @torch.no_grad()
    def search(self, board, current_colour, turn):
        root = Node(0)
        self.last_root = root  # Expose root for Q-target extraction by callers
        
        # Initial expansion
        state_tensor = encode_state(board, current_colour, self.device, out_tensor=self.state_buffer)
        policy_logits, _, q_preds = self.model(state_tensor)
        policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()
        q_values = q_preds[0].cpu().numpy()
        
        valid_moves = get_valid_moves(board, turn)
        
        if len(valid_moves) > 0:
            k = min(20, len(valid_moves))  # top-20: reduces risk of permanently pruning optimal moves
            top_k = np.argsort(policy_probs[valid_moves])[-k:]
            top_k_moves = [valid_moves[idx] for idx in top_k]
            
            policy_sum = sum(policy_probs[a] for a in top_k_moves)
            
            # Add Dirichlet Noise to root node
            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(top_k_moves))
            
            for idx, a in enumerate(top_k_moves):
                p = (1 - epsilon) * (policy_probs[a] / policy_sum) + epsilon * noise[idx] if policy_sum > 0 else 1.0 / len(top_k_moves)
                child_node = Node(p, parent=root, action_from_parent=a)
                child_node.q_prior = q_values[a]
                
                root.children_priors[a] = p
                root.children_q_priors[a] = q_values[a]
                root.children_exists[a] = True
                root.children_nodes[a] = child_node
            root.is_expanded = True
                
        for _ in range(self.num_simulations):
            node = root
            sim_board = fast_clone_board(board)
            sim_colour = current_colour
            sim_turn = turn
            is_terminal = False
            winner = None
            
            # 1. Select
            while node.is_expanded and not is_terminal:
                mask = node.children_exists
                visits = node.children_visits
                
                q = np.zeros(NUM_ACTIONS, dtype=np.float32)
                visited = visits > 0
                q[visited] = node.children_values[visited] / visits[visited]
                q[~visited] = node.children_q_priors[~visited]
                
                sqrt_n = math.sqrt(max(1, node.visit_count))
                u = self.c_puct * node.children_priors * sqrt_n / (1 + visits)
                
                scores = q + u
                scores[~mask] = -np.inf
                
                best_action = int(np.argmax(scores))
                node = node.children_nodes[best_action]
                
                sim_board, sim_colour, sim_turn, is_terminal, winner = apply_move(
                    sim_board, sim_colour, sim_turn, best_action, inplace=True, check_win=False
                )
                
            # 2. Evaluate and Expand
            if is_terminal:
                value = -1.0 
            else:
                state_tensor = encode_state(sim_board, sim_colour, self.device, out_tensor=self.state_buffer)
                policy_logits, value_pred, q_preds = self.model(state_tensor)
                value = value_pred.item()
                policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()
                q_values = q_preds[0].cpu().numpy()
                
                valid_moves = get_valid_moves(sim_board, sim_turn)
                if len(valid_moves) == 0:
                    value = 0 
                else:
                    k = min(20, len(valid_moves))  # top-20 to match root expansion
                    top_k = np.argsort(policy_probs[valid_moves])[-k:]
                    top_k_moves = [valid_moves[idx] for idx in top_k]
                    
                    policy_sum = sum(policy_probs[a] for a in top_k_moves)
                    for a in top_k_moves:
                        p = policy_probs[a] / policy_sum if policy_sum > 0 else 1.0 / len(top_k_moves)
                        child_node = Node(p, parent=node, action_from_parent=a)
                        child_node.q_prior = q_values[a]
                        
                        node.children_priors[a] = p
                        node.children_q_priors[a] = q_values[a]
                        node.children_exists[a] = True
                        node.children_nodes[a] = child_node
                    node.is_expanded = True
                            
            # 3. Backpropagate
            while node is not None:
                node.value_sum += value
                node.visit_count += 1
                
                # Resolve the weak reference to access the actual parent object
                parent = node.parent() if node.parent is not None else None
                
                if parent is not None:
                    parent.children_values[node.action_from_parent] += value
                    parent.children_visits[node.action_from_parent] += 1
                    
                value = -value # Invert value for the other player
                node = parent  # Move up the tree
                
            release_board(sim_board)
                
        # Calculate visit probabilities
        action_probs = torch.zeros(NUM_ACTIONS)
        for a in range(NUM_ACTIONS):
            if root.children_exists[a]:
                action_probs[a] = root.children_visits[a]
            
        if self.temperature == 0:
            best_action = torch.argmax(action_probs)
            action_probs.zero_()
            action_probs[best_action] = 1.0
        else:
            action_probs = action_probs ** (1.0 / self.temperature)
            action_probs /= action_probs.sum()
            
        return action_probs


class BatchedMCTS:
    def __init__(self, model, num_simulations=100, temperature=1.0, c_puct=2.0, add_noise=True):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.device = next(self.model.parameters()).device
        self.board_size = self.model.board_size
        
    @torch.no_grad()
    def search(self, active_games):
        """
        active_games: List of dicts {'board': Board, 'colour': Colour, 'turn': int, 'root': Node}
        """
        num_games = len(active_games)
        if num_games == 0:
            return []
            
        # 1. Expand roots if they have no children
        unexpanded_roots = [i for i, game in enumerate(active_games) if not game['root'].is_expanded]
        if unexpanded_roots:
            state_tensor = torch.zeros(len(unexpanded_roots), 3, self.board_size, self.board_size, device=self.device)
            for idx_idx, idx in enumerate(unexpanded_roots):
                game = active_games[idx]
                encode_state(game['board'], game['colour'], self.device, out_tensor=state_tensor[idx_idx:idx_idx+1])
                
            policy_logits, _, q_preds = self.model(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
            q_values = q_preds.cpu().numpy()
            
            epsilon = 0.25
            alpha = 0.3
            
            for idx_idx, idx in enumerate(unexpanded_roots):
                game = active_games[idx]
                root = game['root']
                valid_moves = get_valid_moves(game['board'], game['turn'])
                
                if valid_moves:
                    p_probs = policy_probs[idx_idx]
                    q_vals = q_values[idx_idx]
                    
                    k = min(20, len(valid_moves))  # top-20 to match root expansion
                    top_k = np.argsort(p_probs[valid_moves])[-k:]
                    top_k_moves = [valid_moves[idx] for idx in top_k]
                    
                    policy_sum = sum(p_probs[a] for a in top_k_moves)
                    if self.add_noise:
                        noise = np.random.dirichlet([alpha] * len(top_k_moves))
                    
                    for noise_idx, a in enumerate(top_k_moves):
                        if self.add_noise:
                            p = (1 - epsilon) * (p_probs[a] / policy_sum) + epsilon * noise[noise_idx] if policy_sum > 0 else 1.0 / len(top_k_moves)
                        else:
                            p = p_probs[a] / policy_sum if policy_sum > 0 else 1.0 / len(top_k_moves)
                        child = Node(p, parent=root, action_from_parent=a)
                        child.q_prior = q_vals[a]
                        
                        root.children_priors[a] = p
                        root.children_q_priors[a] = q_vals[a]
                        root.children_exists[a] = True
                        root.children_nodes[a] = child
                    root.is_expanded = True

        # 2. Simulation Loop
        for _ in range(self.num_simulations):
            search_paths = []
            sim_boards = []
            sim_colours = []
            sim_turns = []
            is_terminals = []
            winners = []
            
            # 2.1 Selection for all games
            for game in active_games:
                node = game['root']
                sim_board = fast_clone_board(game['board'])
                sim_colour = game['colour']
                sim_turn = game['turn']
                is_terminal = False
                winner = None
                
                path = [node]
                while node.is_expanded and not is_terminal:
                    mask = node.children_exists
                    visits = node.children_visits
                    
                    q = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    visited = visits > 0
                    q[visited] = node.children_values[visited] / visits[visited]
                    q[~visited] = node.children_q_priors[~visited]
                    
                    sqrt_n = math.sqrt(max(1, node.visit_count))
                    u = self.c_puct * node.children_priors * sqrt_n / (1 + visits)
                    
                    scores = q + u
                    scores[~mask] = -np.inf
                    
                    best_action = int(np.argmax(scores))
                    node = node.children_nodes[best_action]
                    path.append(node)
                    sim_board, sim_colour, sim_turn, is_terminal, winner = apply_move(
                        sim_board, sim_colour, sim_turn, best_action, inplace=True, check_win=False
                    )
                
                search_paths.append(path)
                sim_boards.append(sim_board)
                sim_colours.append(sim_colour)
                sim_turns.append(sim_turn)
                is_terminals.append(is_terminal)
                winners.append(winner)

            # 2.2 Batched GPU Evaluation
            unexpanded_indices = [i for i, terminal in enumerate(is_terminals) if not terminal]
            
            if unexpanded_indices:
                state_tensor = torch.zeros(len(unexpanded_indices), 3, self.board_size, self.board_size, device=self.device)
                for idx_idx, idx in enumerate(unexpanded_indices):
                    encode_state(sim_boards[idx], sim_colours[idx], self.device, out_tensor=state_tensor[idx_idx:idx_idx+1])
                    
                policy_logits, value_preds, q_preds = self.model(state_tensor)
                policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
                value_preds = value_preds.cpu().numpy()
                q_values = q_preds.cpu().numpy()

            # 2.3 Expansion and Backpropagation
            eval_idx = 0
            for i in range(num_games):
                path = search_paths[i]
                leaf_node = path[-1]
                
                if is_terminals[i]:
                    value = -1.0
                else:
                    value = value_preds[eval_idx][0]
                    p_probs = policy_probs[eval_idx]
                    q_vals = q_values[eval_idx]
                    
                    valid_moves = get_valid_moves(sim_boards[i], sim_turns[i])
                    if not valid_moves:
                        value = 0.0
                    else:
                        k = min(20, len(valid_moves))  # top-20 to match root expansion
                        top_k = np.argsort(p_probs[valid_moves])[-k:]
                        top_k_moves = [valid_moves[idx] for idx in top_k]
                        
                        policy_sum = sum(p_probs[a] for a in top_k_moves)
                        for a in top_k_moves:
                            p = p_probs[a] / policy_sum if policy_sum > 0 else 1.0 / len(top_k_moves)
                            child = Node(p, parent=leaf_node, action_from_parent=a)
                            child.q_prior = q_vals[a]
                            
                            leaf_node.children_priors[a] = p
                            leaf_node.children_q_priors[a] = q_vals[a]
                            leaf_node.children_exists[a] = True
                            leaf_node.children_nodes[a] = child
                        leaf_node.is_expanded = True
                            
                    eval_idx += 1
                    
                # Backpropagate
                for node in reversed(path):
                    node.value_sum += value
                    node.visit_count += 1
                    
                    # Resolve the weak reference
                    parent = node.parent() if node.parent is not None else None
                    
                    if parent is not None:
                        parent.children_values[node.action_from_parent] += value
                        parent.children_visits[node.action_from_parent] += 1
                        
                    value = -value
                    
            for sim_board in sim_boards:
                release_board(sim_board)
                    
        # 3. Final Action Selection Probs
        batch_pis = []
        for i in range(num_games):
            action_probs = torch.zeros(NUM_ACTIONS)
            root = active_games[i]['root']
            for a in range(NUM_ACTIONS):
                if root.children_exists[a]:
                    action_probs[a] = root.children_visits[a]
                
            if self.temperature == 0:
                best_action = torch.argmax(action_probs)
                action_probs.zero_()
                action_probs[best_action] = 1.0
            else:
                action_probs = action_probs ** (1.0 / self.temperature)
                action_probs /= action_probs.sum()
                
            batch_pis.append(action_probs)
            
        return batch_pis
