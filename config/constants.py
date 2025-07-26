"""
Game constants and immutable values for SlytherNN.
"""

from typing import List, Tuple

# Game actions (direction vectors)
ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # DOWN
    (0, -1),  # UP  
    (-1, 0),  # LEFT
    (1, 0)    # RIGHT
]

# Action names for logging/debugging
ACTION_NAMES: List[str] = ["DOWN", "UP", "LEFT", "RIGHT"]

# Game states
GAME_STATES = {
    "RUNNING": 0,
    "GAME_OVER": 1,
    "PAUSED": 2
}

# Reward values
REWARDS = {
    "FOOD": 10.0,
    "DEATH": -10.0,
    "STEP": -0.01,  # Small negative reward for each step
    "WALL": -10.0,
    "SELF": -10.0
}

# Model architecture constants
MODEL_DEFAULTS = {
    "HIDDEN_DIMS": [256, 256],
    "ACTIVATION": "relu",
    "DROPOUT": 0.1
}

# File extensions and patterns
FILE_PATTERNS = {
    "CHECKPOINT": "dqn_snake_checkpoint_ep*.pth",
    "LOG": "training_log*.csv",
    "PLOT": "*.png"
}

# Version info
PROJECT_VERSION = "2.0.0"
PROJECT_NAME = "SlytherNN"
PROJECT_DESCRIPTION = "Deep Q-Network Snake AI with Professional Architecture"

