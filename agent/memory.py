"""
Experience replay memory implementations for SlytherNN.
Includes both basic and prioritized experience replay for different use cases.
"""

from collections import deque
import random
import numpy as np
from typing import List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ReplayMemory:
    """
    Basic experience replay memory with uniform sampling.
    Good for simple DQN implementations and debugging.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay memory.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        logger.debug(f"Initialized ReplayMemory with capacity {max_size}")

    def add(self, experience: Tuple[Any, ...]) -> None:
        """Add experience tuple to memory."""
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """
        Sample batch of experiences uniformly at random.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experience tuples
            
        Raises:
            ValueError: If batch_size > memory size
        """
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} experiences from memory of size {len(self.memory)}")
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return current memory size."""
        return len(self.memory)

    def clear(self) -> None:
        """Clear all stored experiences."""
        self.memory.clear()
        logger.debug("ReplayMemory cleared")

    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self.memory) == self.max_size


class SumTree:
    """
    Binary tree data structure for efficient priority-based sampling.
    Parent node value equals sum of children - enables O(log n) operations.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize SumTree.
        
        Args:
            capacity: Maximum number of leaf nodes (experiences)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float) -> None:
        """
        Update priority of leaf node and propagate change.
        
        Args:
            idx: Tree index of leaf node
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: Any) -> None:
        """
        Add new experience with given priority.
        
        Args:
            priority: Priority value for sampling
            data: Experience data to store
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        """Recursively retrieve leaf index for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience for given cumulative priority sum.
        
        Args:
            s: Cumulative sum value
            
        Returns:
            Tuple of (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        """Get total priority sum (root node value)."""
        return self.tree[0]


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) memory implementation.
    Samples experiences based on TD-error priorities with importance sampling correction.
    
    Based on: "Prioritized Experience Replay" (Schaul et al., 2016)
    """
    
    def __init__(
        self, 
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-4,
        eps: float = 1e-6
    ):
        """
        Initialize prioritized replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment_per_sampling: Beta increment per sampling step
            eps: Small constant to prevent zero priorities
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps
        self.max_priority = 1.0
        
        logger.info(f"Initialized PrioritizedReplayMemory: capacity={capacity}, alpha={alpha}, beta={beta}")

    def add(self, transition: Tuple[Any, ...]) -> None:
        """
        Add experience with maximum priority.
        
        Args:
            transition: Experience tuple (state, action, reward, next_state, done)
        """
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample batch of experiences using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (batch, indices, importance_weights)
        """
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []
        
        # Increment beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # Sample from each segment
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total
        is_weights = np.power(
            self.tree.n_entries * sampling_probabilities + self.eps, 
            -self.beta
        )
        # Normalize weights by maximum weight
        is_weights /= is_weights.max()
        
        return batch, idxs, is_weights

    def update_priorities(self, idxs: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for given indices.
        
        Args:
            idxs: Tree indices to update
            priorities: New priority values (typically TD-errors)
        """
        for idx, priority in zip(idxs, priorities):
            # Add epsilon and take absolute value for numerical stability
            priority = np.abs(priority) + self.eps
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Return current memory size."""
        return self.tree.n_entries

    def get_stats(self) -> dict:
        """Get memory statistics for monitoring."""
        return {
            "size": len(self),
            "capacity": self.tree.capacity,
            "max_priority": self.max_priority,
            "total_priority": self.tree.total,
            "beta": self.beta,
            "alpha": self.alpha
        }

