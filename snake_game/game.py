"""
Enhanced Snake Game implementation with improved state representation for AI learning.
Features better reward shaping, danger detection, and curriculum learning support.
"""

import pygame
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass

from snake_game.utils import random_position

logger = logging.getLogger(__name__)


@dataclass
class GameMetrics:
    """Container for game performance metrics."""
    score: int = 0
    steps: int = 0
    food_eaten: int = 0
    walls_hit: int = 0
    self_collisions: int = 0
    avg_distance_to_food: float = 0.0
    efficiency_ratio: float = 0.0  # food_eaten / steps


class EnhancedSnake:
    """Enhanced Snake with better collision detection and movement tracking."""
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        center = grid_size // 2
        self.body = [
            (center, center),
            (center - 1, center), 
            (center - 2, center)
        ]
        self.direction = (1, 0)  # Start moving right
        self.grow = False
        self.last_move = None
        
    def move(self) -> None:
        """Move snake with movement tracking."""
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        self.last_move = self.direction
        
        if self.grow:
            self.body = [new_head] + self.body
            self.grow = False
        else:
            self.body = [new_head] + self.body[:-1]

    def set_direction(self, dir_tuple: Tuple[int, int]) -> bool:
        """Set direction with validation. Returns True if direction changed."""
        dx, dy = dir_tuple
        # Prevent snake from reversing into itself
        if (dx, dy) == (-self.direction[0], -self.direction[1]):
            return False
        
        old_direction = self.direction
        self.direction = dir_tuple
        return old_direction != dir_tuple

    def grow_snake(self) -> None:
        """Mark snake for growth on next move."""
        self.grow = True

    def head(self) -> Tuple[int, int]:
        """Get snake head position."""
        return self.body[0]

    def collided_with_self(self) -> bool:
        """Check if snake head collides with body."""
        return self.body[0] in self.body[1:]

    def collided_with_wall(self) -> bool:
        """Check if snake head is outside grid boundaries."""
        x, y = self.body[0]
        return x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size
    
    def get_danger_directions(self) -> Dict[str, bool]:
        """Get danger information for each direction."""
        x, y = self.head()
        dangers = {}
        
        # Check each possible direction
        directions = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        
        for name, (dx, dy) in directions.items():
            new_x, new_y = x + dx, y + dy
            
            # Wall collision
            wall_danger = (new_x < 0 or new_y < 0 or 
                          new_x >= self.grid_size or new_y >= self.grid_size)
            
            # Self collision
            self_danger = (new_x, new_y) in self.body
            
            dangers[name] = wall_danger or self_danger
        
        return dangers
    
    def distance_to_tail(self) -> int:
        """Get distance from head to tail."""
        if len(self.body) < 2:
            return 0
        head = self.head()
        tail = self.body[-1]
        return abs(head[0] - tail[0]) + abs(head[1] - tail[1])


class EnhancedFruit:
    """Enhanced fruit with positioning strategies."""
    
    def __init__(self, grid_size: int, snake_body: List[Tuple[int, int]], strategy: str = "random"):
        self.grid_size = grid_size
        self.strategy = strategy
        self.position = self._new_position(snake_body)
        self.spawn_count = 0

    def _new_position(self, snake_body: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Generate new fruit position with different strategies."""
        self.spawn_count += 1
        
        if self.strategy == "random":
            return self._random_position(snake_body)
        elif self.strategy == "avoid_corners":
            return self._avoid_corners_position(snake_body)
        elif self.strategy == "center_bias":
            return self._center_bias_position(snake_body)
        else:
            return self._random_position(snake_body)
    
    def _random_position(self, snake_body: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Original random positioning."""
        attempts = 0
        while attempts < 100:  # Prevent infinite loops
            pos = random_position(self.grid_size)
            if pos not in snake_body:
                return pos
            attempts += 1
        
        # Fallback: find any empty position
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in snake_body:
                    return (x, y)
        
        # Should never reach here in normal gameplay
        return (0, 0)
    
    def _avoid_corners_position(self, snake_body: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Avoid placing fruit in corners (easier for AI)."""
        attempts = 0
        corner_margin = 2
        
        while attempts < 100:
            x = np.random.randint(corner_margin, self.grid_size - corner_margin)
            y = np.random.randint(corner_margin, self.grid_size - corner_margin)
            pos = (x, y)
            
            if pos not in snake_body:
                return pos
            attempts += 1
        
        # Fallback to random
        return self._random_position(snake_body)
    
    def _center_bias_position(self, snake_body: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Bias fruit towards center (curriculum learning)."""
        center = self.grid_size // 2
        attempts = 0
        
        while attempts < 100:
            # Gaussian distribution around center
            x = int(np.clip(np.random.normal(center, self.grid_size // 4), 0, self.grid_size - 1))
            y = int(np.clip(np.random.normal(center, self.grid_size // 4), 0, self.grid_size - 1))
            pos = (x, y)
            
            if pos not in snake_body:
                return pos
            attempts += 1
        
        return self._random_position(snake_body)

    def respawn(self, snake_body: List[Tuple[int, int]]) -> None:
        """Respawn fruit at new position."""
        self.position = self._new_position(snake_body)


class EnhancedSnakeGame:
    """
    Enhanced Snake game with superior AI training features:
    - Better state representation with danger detection
    - Improved reward shaping  
    - Curriculum learning support
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self, 
        grid_size: int = 12, 
        cell_size: int = 32, 
        mode: str = "ai",
        fruit_strategy: str = "random",
        max_steps: int = 500
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.mode = mode
        self.max_steps = max_steps
        
        # Game entities
        self.snake = EnhancedSnake(grid_size)
        self.fruit = EnhancedFruit(grid_size, self.snake.body, fruit_strategy)
        
        # Game state
        self.running = True
        self.won = False
        self.step_count = 0
        
        # Metrics tracking
        self.metrics = GameMetrics()
        self.distance_history = []
        
        logger.debug(f"EnhancedSnakeGame initialized: {grid_size}x{grid_size}, mode={mode}")

    def reset(self) -> None:
        """Reset game to initial state."""
        self.snake = EnhancedSnake(self.grid_size)
        self.fruit = EnhancedFruit(self.grid_size, self.snake.body, self.fruit.strategy)
        self.running = True
        self.won = False
        self.step_count = 0
        self.metrics = GameMetrics()
        self.distance_history = []
        logger.debug("EnhancedSnakeGame reset")

    def update(self) -> None:
        """Update game state for one step."""
        if not self.running:
            return
        
        self.step_count += 1
        self.metrics.steps += 1
        
        # Track distance to food before moving
        distance_to_food = self._manhattan_distance(self.snake.head(), self.fruit.position)
        self.distance_history.append(distance_to_food)
        
        self.snake.move()
        
        # Check fruit collection
        if self.snake.head() == self.fruit.position:
            self.snake.grow_snake()
            self.metrics.score += 1
            self.metrics.food_eaten += 1
            
            # Check win condition
            if self._check_win_condition():
                self.won = True
                self.running = False
                logger.info(f"Perfect game achieved! Score: {self.metrics.score}")
                return
            
            # Respawn fruit
            self.fruit.respawn(self.snake.body)
            
        # Check collision conditions
        if self.snake.collided_with_wall():
            self.metrics.walls_hit += 1
            self.running = False
            logger.debug(f"Wall collision. Final score: {self.metrics.score}")
        elif self.snake.collided_with_self():
            self.metrics.self_collisions += 1
            self.running = False
            logger.debug(f"Self collision. Final score: {self.metrics.score}")
        
        # Check step limit
        if self.step_count >= self.max_steps:
            self.running = False
            logger.debug(f"Step limit reached. Final score: {self.metrics.score}")
        
        # Update efficiency metrics
        if self.distance_history:
            self.metrics.avg_distance_to_food = np.mean(self.distance_history)
        
        if self.metrics.steps > 0:
            self.metrics.efficiency_ratio = self.metrics.food_eaten / self.metrics.steps

    def step(self, action_idx: int, device: torch.device) -> Tuple[torch.Tensor, float, bool]:
        """Enhanced step function with intelligent reward shaping."""
        from enhanced_dqn_agent import ACTIONS
        
        if not isinstance(action_idx, int) or action_idx not in range(len(ACTIONS)):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        # Store previous state for reward calculation
        prev_head = self.snake.head()
        prev_distance = self._manhattan_distance(prev_head, self.fruit.position)
        prev_score = self.metrics.score
        
        # Apply action
        direction_changed = self.snake.set_direction(ACTIONS[action_idx])
        self.update()
        
        # Calculate reward
        reward = self._calculate_enhanced_reward(
            prev_head, prev_distance, prev_score, direction_changed
        )
        
        # Get next state
        next_state = self.get_enhanced_state(device)
        done = not self.running
        
        return next_state, reward, done

    def _calculate_enhanced_reward(
        self, 
        prev_head: Tuple[int, int], 
        prev_distance: int, 
        prev_score: int,
        direction_changed: bool
    ) -> float:
        """Calculate enhanced reward with multiple components."""
        reward = 0.0
        
        # 1. Terminal rewards
        if not self.running:
            if self.won:
                return 100.0  # Perfect game bonus
            elif self.snake.collided_with_wall():
                return -15.0  # Wall penalty
            elif self.snake.collided_with_self():
                return -20.0  # Self collision penalty (worse than wall)
            else:
                return -5.0   # Time limit penalty
        
        # 2. Food reward
        if self.metrics.score > prev_score:
            reward += 10.0 + (self.metrics.score * 0.5)  # Increasing reward for longer snakes
            return reward  # Early return for food reward
        
        # 3. Distance-based reward (potential shaping)
        current_distance = self._manhattan_distance(self.snake.head(), self.fruit.position)
        distance_reward = (prev_distance - current_distance) * 0.1
        reward += distance_reward
        
        # 4. Survival reward (small positive for staying alive)
        reward += 0.01
        
        # 5. Danger penalties
        dangers = self.snake.get_danger_directions()
        current_direction_name = self._get_direction_name(self.snake.direction)
        
        # Penalty for moving toward immediate danger
        if dangers.get(current_direction_name, False):
            reward -= 0.5
        
        # Bonus for avoiding danger when close
        danger_count = sum(dangers.values())
        if danger_count >= 3:  # Surrounded
            reward -= 1.0
        elif danger_count == 0:  # Safe
            reward += 0.1
        
        # 6. Movement efficiency
        # Penalty for not changing direction when it would be beneficial
        if not direction_changed and current_distance > prev_distance:
            reward -= 0.05
        
        # 7. Space utilization (encourage exploring available space)
        head_x, head_y = self.snake.head()
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        distance_from_center = abs(head_x - center_x) + abs(head_y - center_y)
        max_distance_from_center = self.grid_size - 1
        
        # Small bonus for staying away from edges when snake is small
        if len(self.snake.body) < self.grid_size:
            edge_distance = min(head_x, head_y, self.grid_size - 1 - head_x, self.grid_size - 1 - head_y)
            if edge_distance <= 1:
                reward -= 0.2  # Penalty for being too close to edges
        
        return reward

    def _get_direction_name(self, direction: Tuple[int, int]) -> str:
        """Convert direction tuple to name."""
        direction_map = {
            (0, -1): 'up',
            (0, 1): 'down',
            (-1, 0): 'left',
            (1, 0): 'right'
        }
        return direction_map.get(direction, 'unknown')

    def get_enhanced_state(self, device: torch.device) -> torch.Tensor:
        """
        Get enhanced state representation with danger detection and spatial awareness.
        
        State components:
        1. Grid representation (flattened)
        2. Direction one-hot encoding
        3. Relative food position (normalized)
        4. Danger detection (4 directions)
        5. Snake length (normalized)
        6. Distance to walls (4 directions, normalized)
        7. Food distance (normalized)
        """
        
        # 1. Grid encoding (snake body=1.0, head=2.0, fruit=3.0, empty=0.0)
        grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32, device=device)
        
        # Encode snake body
        for i, (x, y) in enumerate(self.snake.body):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if i == 0:  # Head
                    grid[y, x] = 2.0
                else:  # Body
                    grid[y, x] = 1.0
        
        # Encode fruit
        fx, fy = self.fruit.position
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            grid[fy, fx] = 3.0

        # 2. Direction one-hot encoding
        direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        direction_vector = torch.zeros(4, device=device)
        direction_vector[direction_map.get(self.snake.direction, 0)] = 1.0

        # 3. Relative food position (normalized)
        head_x, head_y = self.snake.head()
        rel_food_x = (fx - head_x) / (self.grid_size - 1)
        rel_food_y = (fy - head_y) / (self.grid_size - 1)
        rel_food = torch.tensor([rel_food_x, rel_food_y], dtype=torch.float32, device=device)

        # 4. Danger detection (one-hot for each direction)
        dangers = self.snake.get_danger_directions()
        danger_vector = torch.tensor([
            float(dangers.get('up', False)),
            float(dangers.get('down', False)),
            float(dangers.get('left', False)),
            float(dangers.get('right', False))
        ], dtype=torch.float32, device=device)

        # 5. Snake length (normalized)
        snake_length = torch.tensor([len(self.snake.body) / (self.grid_size * self.grid_size)], 
                                   dtype=torch.float32, device=device)

        # 6. Distance to walls (normalized)
        wall_distances = torch.tensor([
            head_y / (self.grid_size - 1),                    # Distance to top
            (self.grid_size - 1 - head_y) / (self.grid_size - 1),  # Distance to bottom
            head_x / (self.grid_size - 1),                    # Distance to left
            (self.grid_size - 1 - head_x) / (self.grid_size - 1)   # Distance to right
        ], dtype=torch.float32, device=device)

        # 7. Food distance (normalized)
        food_distance = torch.tensor([
            self._manhattan_distance(self.snake.head(), self.fruit.position) / (self.grid_size * 2)
        ], dtype=torch.float32, device=device)

        # Concatenate all features
        flat_grid = grid.flatten()
        enhanced_state = torch.cat([
            flat_grid,           # Grid representation
            direction_vector,    # Current direction
            rel_food,           # Relative food position
            danger_vector,      # Danger detection
            snake_length,       # Snake length
            wall_distances,     # Wall distances
            food_distance       # Food distance
        ])
        
        return enhanced_state

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_win_condition(self) -> bool:
        """Check if snake has achieved perfect game."""
        max_possible_length = self.grid_size * self.grid_size
        current_length = len(self.snake.body)
        if self.snake.grow:
            current_length += 1
        return current_length >= max_possible_length

    def get_game_info(self) -> Dict:
        """Get comprehensive game information for analysis."""
        return {
            'snake_head': self.snake.head(),
            'snake_body': self.snake.body.copy(),
            'snake_length': len(self.snake.body),
            'food_pos': self.fruit.position,
            'grid_size': self.grid_size,
            'running': self.running,
            'won': self.won,
            'step_count': self.step_count,
            'metrics': self.metrics,
            'dangers': self.snake.get_danger_directions(),
            'distance_to_food': self._manhattan_distance(self.snake.head(), self.fruit.position)
        }

    # Pygame rendering methods (kept for compatibility)
    def draw(self, screen: pygame.Surface, board_offset_x: int = 0, board_offset_y: int = 0) -> None:
        """Enhanced rendering with danger visualization."""
        # Modern dark gradient background
        for y in range(screen.get_height()):
            intensity = int(40 * (1 - y / screen.get_height()))
            color = (26 + intensity, 26 + intensity, 32 + intensity * 2)
            pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))

        # Draw grid border with glow effect
        grid_w = self.grid_size * self.cell_size
        grid_h = self.grid_size * self.cell_size
        border_rect = pygame.Rect(board_offset_x - 2, board_offset_y - 2, grid_w + 4, grid_h + 4)
        
        # Glow effect
        for i in range(3):
            pygame.draw.rect(screen, (60 + i * 20, 60 + i * 20, 80 + i * 20), 
                           border_rect, width=3 - i, border_radius=12)

        # Draw snake with gradient effect
        for i, segment in enumerate(self.snake.body):
            # Color gradient from head to tail
            if i == 0:  # Head
                color = (0, 255, 100)  # Bright green head
                # Add direction indicator
                head_rect = pygame.Rect(
                    board_offset_x + segment[0] * self.cell_size + 2,
                    board_offset_y + segment[1] * self.cell_size + 2,
                    self.cell_size - 4, self.cell_size - 4
                )
                pygame.draw.rect(screen, color, head_rect, border_radius=8)
                
                # Direction arrow
                center_x = head_rect.centerx
                center_y = head_rect.centery
                dx, dy = self.snake.direction
                arrow_end = (center_x + dx * 8, center_y + dy * 8)
                pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), arrow_end, 3)
                
            else:  # Body
                # Gradient from green to dark green
                intensity = max(100, 255 - i * 10)
                color = (0, intensity, intensity // 3)
                pygame.draw.rect(
                    screen, color,
                    (
                        board_offset_x + segment[0] * self.cell_size + 1,
                        board_offset_y + segment[1] * self.cell_size + 1,
                        self.cell_size - 2, self.cell_size - 2
                    ),
                    border_radius=6
                )
        
        # Draw fruit with pulsing effect
        if not self.won:
            fx, fy = self.fruit.position
            # Pulsing effect based on game step
            pulse = abs((self.step_count % 60) - 30) / 30.0
            base_color = 220
            pulse_color = int(base_color + pulse * 35)
            
            fruit_rect = pygame.Rect(
                board_offset_x + fx * self.cell_size + 2,
                board_offset_y + fy * self.cell_size + 2,
                self.cell_size - 4, self.cell_size - 4
            )
            pygame.draw.ellipse(screen, (pulse_color, 60, 60), fruit_rect)
            
            # Glow effect around fruit
            glow_rect = pygame.Rect(
                board_offset_x + fx * self.cell_size,
                board_offset_y + fy * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.ellipse(screen, (100, 30, 30), glow_rect, width=2)
        
        # Draw enhanced scoreboard
        self.draw_enhanced_scoreboard(screen, board_offset_x, board_offset_y)

    def draw_enhanced_scoreboard(self, screen: pygame.Surface, board_offset_x: int, board_offset_y: int) -> None:
        """Draw enhanced scoreboard with metrics."""
        font_large = pygame.font.SysFont("arial", 28, bold=True)
        font_small = pygame.font.SysFont("arial", 18)
        
        # Main score
        score_text = f"Score: {self.metrics.score}"
        if self.won:
            score_text += " - PERFECT! 🏆"
        
        score_surface = font_large.render(score_text, True, (255, 255, 255))
        screen.blit(score_surface, (board_offset_x, board_offset_y - 40))
        
        # Additional metrics in AI mode
        if self.mode == "ai":
            metrics_y = board_offset_y - 15
            
            # Steps and efficiency
            steps_text = f"Steps: {self.metrics.steps}"
            steps_surface = font_small.render(steps_text, True, (200, 200, 200))
            screen.blit(steps_surface, (board_offset_x, metrics_y))
            
            if self.metrics.steps > 0:
                efficiency = self.metrics.food_eaten / self.metrics.steps
                eff_text = f"Efficiency: {efficiency:.3f}"
                eff_surface = font_small.render(eff_text, True, (200, 200, 200))
                screen.blit(eff_surface, (board_offset_x + 120, metrics_y))
            
            # Distance to food
            if hasattr(self, 'fruit'):
                distance = self._manhattan_distance(self.snake.head(), self.fruit.position)
                dist_text = f"Distance: {distance}"
                dist_surface = font_small.render(dist_text, True, (200, 200, 200))
                screen.blit(dist_surface, (board_offset_x + 250, metrics_y))

    def draw_game_over(self, screen: pygame.Surface) -> None:
        """Draw enhanced game over screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface(screen.get_size())
        overlay.set_alpha(160)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Title
        font_title = pygame.font.SysFont("arial", 48, bold=True)
        font_stats = pygame.font.SysFont("arial", 24)
        font_controls = pygame.font.SysFont("arial", 20)
        
        center_x = screen.get_width() // 2
        
        if self.won:
            title_text = "🏆 PERFECT GAME! 🏆"
            title_color = (0, 255, 100)
        else:
            title_text = "Game Over"
            title_color = (255, 100, 100)
        
        title_surface = font_title.render(title_text, True, title_color)
        title_rect = title_surface.get_rect(center=(center_x, 180))
        screen.blit(title_surface, title_rect)
        
        # Detailed statistics
        stats = [
            f"Final Score: {self.metrics.score}",
            f"Steps Taken: {self.metrics.steps}",
            f"Food Eaten: {self.metrics.food_eaten}",
            f"Efficiency: {self.metrics.efficiency_ratio:.3f}",
            f"Avg Distance to Food: {self.metrics.avg_distance_to_food:.1f}"
        ]
        
        y_offset = 240
        for stat in stats:
            stat_surface = font_stats.render(stat, True, (255, 255, 255))
            stat_rect = stat_surface.get_rect(center=(center_x, y_offset))
            screen.blit(stat_surface, stat_rect)
            y_offset += 30
        
        # Controls
        controls = [
            "Press R to Restart",
            "Press ESC to Return to Menu"
        ]
        
        if self.mode == "ai":
            controls.insert(1, "Press SPACE for AI Analysis")
        
        y_offset += 20
        for control in controls:
            control_surface = font_controls.render(control, True, (200, 200, 200))
            control_rect = control_surface.get_rect(center=(center_x, y_offset))
            screen.blit(control_surface, control_rect)
            y_offset += 25


class CurriculumSnakeGame(EnhancedSnakeGame):
    """Snake game variant for curriculum learning with progressive difficulty."""
    
    def __init__(self, stage: int = 0, **kwargs):
        self.stage = stage
        
        # Curriculum stages with progressive difficulty
        stage_configs = [
            {'grid_size': 8, 'max_steps': 200, 'fruit_strategy': 'center_bias'},    # Stage 0: Easy
            {'grid_size': 10, 'max_steps': 300, 'fruit_strategy': 'avoid_corners'}, # Stage 1: Medium
            {'grid_size': 12, 'max_steps': 400, 'fruit_strategy': 'random'},        # Stage 2: Normal  
            {'grid_size': 12, 'max_steps': 500, 'fruit_strategy': 'random'},        # Stage 3: Extended
        ]
        
        # Override kwargs with stage config
        if stage < len(stage_configs):
            config = stage_configs[stage]
            kwargs.update(config)
        
        super().__init__(**kwargs)
        logger.info(f"Curriculum stage {stage} initialized: {kwargs}")