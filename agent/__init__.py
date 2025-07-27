"""
Agent modules for SlytherNN - Neural networks and experience replay.
"""

from .dqn import DQN, ACTIONS
from .memory import ReplayMemory, PrioritizedReplayMemory

__all__ = [
    "DQN",
    "ACTIONS", 
    "ReplayMemory",
    "PrioritizedReplayMemory"
]

