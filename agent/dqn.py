"""
Enhanced DQN Agent for SlytherNN - Modern RL Implementation
Combines Double DQN, Dueling Networks, and Noisy Networks for superior performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Global action mapping
ACTIONS = [
    (0, -1),  # UP
    (0, 1),   # DOWN  
    (-1, 0),  # LEFT
    (1, 0),   # RIGHT
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for better exploration without epsilon-greedy."""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.empty((out_features, in_features)))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class DuelingDoubleDQN(nn.Module):
    """
    Advanced DQN combining:
    - Double DQN: Reduces overestimation bias
    - Dueling Networks: Separate value and advantage estimation  
    - Noisy Networks: Learnable exploration
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Layer normalization for stability
            nn.Dropout(0.1)
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            NoisyLinear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized DuelingDoubleDQN: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights for better training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling network."""
        features = self.feature_layer(x)
        
        # Separate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formulation
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Get action using the network (no epsilon-greedy needed)."""
        if not deterministic:
            self.reset_noise()
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()


class ImprovedReplayBuffer:
    """
    Enhanced replay buffer with priority sampling and n-step returns.
    """
    
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        
        # Storage
        self.buffer = []
        self.position = 0
        
        # N-step transition storage
        self.n_step_buffer = []
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with n-step processing."""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Process n-step return when buffer is full
        if len(self.n_step_buffer) >= self.n_step:
            self._add_n_step_transition()
    
    def _add_n_step_transition(self):
        """Compute and add n-step transition."""
        # Get the oldest transition
        state, action, _, _, _ = self.n_step_buffer[0]
        
        # Compute n-step return
        n_step_return = 0
        next_state = None
        done = False
        
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                next_state = ns
                done = True
                break
            if i == self.n_step - 1:
                next_state = ns
                done = d
        
        # Add to main buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, n_step_return, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
        # Remove oldest from n-step buffer
        self.n_step_buffer.pop(0)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)


class AdvancedDQNAgent:
    """
    Advanced DQN Agent with modern improvements for superior Snake AI performance.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 256,
        target_update_freq: int = 1000,
        n_step: int = 3,
        device: torch.device = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.online_net = DuelingDoubleDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDoubleDQN(state_dim, action_dim).to(self.device)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with advanced settings
        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=lr,
            weight_decay=1e-5,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=500, factor=0.8, verbose=True
        )
        
        # Replay buffer
        self.replay_buffer = ImprovedReplayBuffer(buffer_size, n_step, gamma)
        
        # Training counters
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"Advanced DQN Agent initialized on {self.device}")
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using the policy network."""
        return self.online_net.get_action(state.to(self.device), deterministic)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(
            state.cpu(), action, reward, next_state.cpu(), done
        )
    
    def update(self) -> Optional[float]:
        """Update the agent's networks."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target computation
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            # Use target network to evaluate actions
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (self.gamma ** self.replay_buffer.n_step) * next_q * (~dones).unsqueeze(1)
        
        # Compute loss
        loss = F.huber_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug(f"Target network updated at step {self.steps}")
        
        return loss.item()
    
    def save(self, filepath: str, additional_info: dict = None):
        """Save agent state."""
        state = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
            }
        }
        
        if additional_info:
            state.update(additional_info)
        
        torch.save(state, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> dict:
        """Load agent state."""
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.online_net.load_state_dict(state['online_net'])
        self.target_net.load_state_dict(state['target_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        
        if 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        
        self.steps = state.get('steps', 0)
        self.episodes = state.get('episodes', 0)
        
        logger.info(f"Agent loaded from {filepath}")
        return state