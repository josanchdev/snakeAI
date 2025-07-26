"""
Configuration module for SlytherNN.
Provides centralized configuration management for all project components.
"""

from .settings import GameConfig, TrainingConfig, UIConfig, ProjectConfig
from .constants import ACTIONS, ACTION_NAMES, REWARDS, MODEL_DEFAULTS

__all__ = [
    "GameConfig",
    "TrainingConfig", 
    "UIConfig",
    "ProjectConfig",
    "ACTIONS",
    "ACTION_NAMES", 
    "REWARDS",
    "MODEL_DEFAULTS"
]

