import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.cem.mcts import MCTS


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class Hex3HNN(nn.Module):
    """
    A 3-Head Convolutional Neural Network architecture for learning Hex.
    Takes 2D spatial board state as input and outputs:
    1. Policy (action probabilities)
    2. State-Value (win probability for current player)
    3. Action-Value (Q-values for each possible action)
    """
    def __init__(self, board_size: int = 11, temperature: float = 0.1, num_resBlocks: int = 4, num_hidden: int = 64):
        super(Hex3HNN, self).__init__()
        self.board_size = board_size
        self.temperature = temperature

        # Initial Convolution to process input channels
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # ResNet Backbone preserves spatial 2D structure
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        
        # 1. Policy Head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        )
        
        # 2. State-Value Head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # 3. Action-Value Head (Q-values)
        self.actionValueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
            
        policy_logits = self.policyHead(x)
        value = self.valueHead(x)
        action_value = self.actionValueHead(x)
        
        return policy_logits, value, action_value


class Agent1(AgentBase):
    def __init__(self, colour: Colour, board_size: int = None, temperature: float = 0.1):
        super().__init__(colour)
        self.temperature = temperature

        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine board size
        if board_size is None:
            # Try to infer from checkpoint
            model_path = os.path.join(os.path.dirname(__file__), "checkpoints/best_model.pt")
            if os.path.isfile(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    if 'board_size' in checkpoint:
                        board_size = checkpoint['board_size']
                except Exception:
                    pass
        
        # Final fallback to 11
        self.board_size = board_size if board_size is not None else 11
        
        # Initialize the 3HNN architecture and move to device
        self.model = Hex3HNN(board_size=self.board_size, temperature=self.temperature).to(self.device)
        self.model.eval()  # Default to evaluation mode
        
        # Load best model if exists
        model_path = os.path.join(os.path.dirname(__file__), "checkpoints/best_model.pt")
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Predicts the best move using MCTS guided by the 3HNN model.
        """
        # We use a relatively small number of simulations for tournament play to stay within time limits
        mcts = MCTS(self.model, num_simulations=100, temperature=self.temperature) # low temp for greedy play
        
        # MCTS handles board encoding internally
        action_probs = mcts.search(board, self.colour, turn)
        
        # Select best action
        best_action = torch.argmax(action_probs).item()
        
        if best_action == self.board_size * self.board_size:
            return Move(-1, -1) # Swap move
            
        x, y = divmod(best_action, self.board_size)
        return Move(x, y)