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
    """Enhanced training configuration for superior AI performance."""
    
    # Environment settings - optimized for your RTX 3090
    num_envs: int = 96
    num_episodes: int = 4000  # More episodes for complex learning
    max_steps_per_episode: int = 300  # Allow longer games
    
    # Neural network settings - upgraded architecture
    learning_rate: float = 2e-4  # Slightly lower for stable convergence
    batch_size: int = 512  # Increased for your RTX 3090
    memory_size: int = 300000  # Larger memory for better sampling
    
    # Advanced DQN settings
    gamma: float = 0.995  
    target_update_freq: int = 1500  # Less frequent updates for stability
    
    # Enhanced exploration - better than simple epsilon decay
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  # Lower final epsilon
    epsilon_decay: float = 0.99975  # Slower decay for better exploration
    
    # Training optimization
    grad_accum_steps: int = 1  # Your RTX 3090 can handle full batches
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0

    # Enhanced checkpointing
    save_every: int = 250  # More frequent saves
    log_every: int = 25   # More detailed logging

    # 🎮 Anti-plateau game settings
    reward_scaling: float = 1.0  # ✅ New: Can adjust reward magnitude
    curriculum_learning: bool = True  # ✅ New: Progressive difficulty

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device and reproducibility
    device: Optional[str] = None
    seed: int = 42


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

