"""
Configuration classes for SlytherNN project.
Centralized settings management for game, training, and UI components.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import torch
from pathlib import Path


@dataclass
class GameConfig:
    """Game mechanics and display configuration."""
    
    # Grid settings
    grid_size: int = 12
    cell_size: int = 32
    
    # Display settings
    screen_width: int = 600
    screen_height: int = 600
    fps: int = 60
    
    # Game timing
    ai_move_delay_ms: int = 90
    
    # Colors (RGB tuples)
    background_color: Tuple[int, int, int] = (26, 26, 32)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    snake_color: Tuple[int, int, int] = (0, 255, 0)
    food_color: Tuple[int, int, int] = (255, 0, 0)
    
    # Fonts
    menu_font_size: int = 36
    game_font_size: int = 24
    font_family: str = "arial"
    
    @property
    def grid_pixel_width(self) -> int:
        """Calculate pixel width of the game grid."""
        return self.grid_size * self.cell_size
    
    @property
    def board_offset_x(self) -> int:
        """Calculate X offset to center the game board."""
        return (self.screen_width - self.grid_pixel_width) // 2
    
    @property
    def board_offset_y(self) -> int:
        """Calculate Y offset to center the game board."""
        return (self.screen_height - self.grid_pixel_width) // 2


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""
    
    # Environment settings
    num_envs: int = 64
    num_episodes: int = 4000
    max_steps_per_episode: int = 100
    
    # Neural network settings
    learning_rate: float = 5e-4
    batch_size: int = 128
    memory_size: int = 100000
    
    # DQN specific settings
    gamma: float = 0.99  # Discount factor
    target_update_freq: int = 1000  # Steps between target network updates
    
    # Exploration settings
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    
    # Training optimization
    grad_accum_steps: int = 2
    use_mixed_precision: bool = True
    
    # Checkpointing and logging
    save_every: int = 1000  # Save every N episodes
    log_every: int = 100   # Log every N episodes
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device and reproducibility
    device: Optional[str] = None
    seed: int = 42
    
    def __post_init__(self):
        """Set device automatically if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class UIConfig:
    """User interface and visualization settings."""
    
    # Menu settings
    show_fps: bool = True
    show_score: bool = True
    show_ai_info: bool = True  # Show Q-values, action confidence
    
    # Visualization settings
    enable_game_recording: bool = False
    recording_fps: int = 30
    
    # Training visualization
    plot_training_progress: bool = True
    update_plot_every: int = 500  # Episodes
    
    # Terminal output
    verbose_logging: bool = True
    progress_bar: bool = True


@dataclass
class ProjectConfig:
    """Overall project configuration container."""
    
    game: GameConfig
    training: TrainingConfig
    ui: UIConfig
    
    @classmethod
    def default(cls) -> 'ProjectConfig':
        """Create default configuration."""
        return cls(
            game=GameConfig(),
            training=TrainingConfig(),
            ui=UIConfig()
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file."""
        import yaml
        
        config_dict = {
            'game': self.game.__dict__,
            'training': self.training.__dict__,
            'ui': self.ui.__dict__
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            game=GameConfig(**config_dict.get('game', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            ui=UIConfig(**config_dict.get('ui', {}))
        )

