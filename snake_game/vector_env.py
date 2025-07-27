"""
Vectorized environment for efficient parallel training.
Runs multiple Snake game instances concurrently for better sample efficiency.
"""

import torch
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor
import logging

from snake_game.game import SnakeGame
from utils import get_device

logger = logging.getLogger(__name__)


class VectorEnv:
    """
    Vectorized environment that runs multiple Snake games in parallel.
    Optimized for efficient RL training with batch operations.
    """
    
    def __init__(
        self, 
        num_envs: int = 64, 
        grid_size: int = 12, 
        cell_size: int = 32, 
        device: torch.device = None
    ):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            grid_size: Size of game grid
            cell_size: Size of each cell in pixels
            device: PyTorch device for tensor operations
        """
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.device = device or get_device()
        
        # Create parallel game environments
        self.envs = [
            SnakeGame(grid_size=grid_size, cell_size=cell_size, mode="ai") 
            for _ in range(num_envs)
        ]
        
        logger.info(f"VectorEnv initialized: {num_envs} environments on {self.device}")

    def reset(self) -> torch.Tensor:
        """
        Reset all environments and return batched initial states.
        
        Returns:
            Batched states tensor with shape (num_envs, state_dim)
        """
        for env in self.envs:
            env.reset()
        return self.get_states()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step all environments with given actions in parallel.
        Automatically resets environments that are done.

        Args:
            actions: Tensor of action indices with shape (num_envs,)

        Returns:
            next_states: Tensor with shape (num_envs, state_dim)
            rewards: Tensor with shape (num_envs,)
            dones: Tensor with shape (num_envs,)
        """
        def step_single_env(env: SnakeGame, action: int) -> Tuple[torch.Tensor, float, bool]:
            """Step a single environment, resetting if needed."""
            if not env.running:
                env.reset()
            return env.step(int(action), self.device)

        # Use ThreadPoolExecutor for parallel stepping
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            results = list(executor.map(step_single_env, self.envs, actions))

        # Unpack results
        next_states, rewards, dones = zip(*results)
        
        return (
            torch.stack(next_states),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.bool)
        )

    def get_states(self) -> torch.Tensor:
        """
        Get current states from all environments as batched tensor.
        
        Returns:
            Batched states tensor with shape (num_envs, state_dim)
        """
        states = [env.get_state(self.device) for env in self.envs]
        return torch.stack(states)

    def all_running(self) -> List[bool]:
        """Get running status of all environments."""
        return [env.running for env in self.envs]
    
    def get_scores(self) -> List[int]:
        """Get current scores from all environments."""
        return [env.score for env in self.envs]
    
    def get_stats(self) -> dict:
        """Get statistics about the vectorized environment."""
        running_count = sum(self.all_running())
        scores = self.get_scores()
        
        return {
            "num_envs": self.num_envs,
            "running_envs": running_count,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0
        }

