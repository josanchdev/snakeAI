"""
Deep Q-Network implementation for SlytherNN.
Defines the neural network architecture and action space for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Global action mapping for all modules - matches your original game.py
ACTIONS: List[Tuple[int, int]] = [
    (0, -1),  # UP
    (0, 1),   # DOWN  
    (-1, 0),  # LEFT
    (1, 0),   # RIGHT
]

# Action names for logging and debugging
ACTION_NAMES: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]


class DQN(nn.Module):
    """
    Deep Q-Network for estimating action-value functions.
    
    Architecture:
    - Input: Game state representation (flattened grid + features)
    - Hidden: 2 fully connected layers with ReLU activation
    - Output: Q-values for each possible action
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize the DQN.
        
        Args:
            input_dim: Size of input state vector
            output_dim: Number of possible actions
            hidden_dim: Size of hidden layers (increased from 128 for better capacity)
        """
        super(DQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for better training stability
        self._init_weights()
        
        logger.debug(f"Initialized DQN: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor, shape (batch_size, input_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            epsilon: Exploration probability
            
        Returns:
            Action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.output_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return torch.argmax(q_values).item()

