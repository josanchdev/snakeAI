"""
Snake Game implementation for SlytherNN - Core game logic and mechanics.
Provides both human and AI gameplay modes with professional rendering.
"""

import pygame
import sys
import numpy as np
import torch
from typing import Tuple, Optional
import logging

from snake_game.utils import random_position

logger = logging.getLogger(__name__)


class Snake:
    """Snake entity with movement, growth, and collision detection."""
    
    def __init__(self, grid_size: int):
        """
        Initialize snake at center of grid.
        
        Args:
            grid_size: Size of the game grid
        """
        self.grid_size = grid_size
        center = grid_size // 2
        self.body = [
            (center, center),
            (center - 1, center), 
            (center - 2, center)
        ]
        self.direction = (1, 0)  # Start moving right
        self.grow = False

    def move(self) -> None:
        """Move snake one step in current direction."""
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        if self.grow:
            self.body = [new_head] + self.body
            self.grow = False
        else:
            self.body = [new_head] + self.body[:-1]

    def set_direction(self, dir_tuple: Tuple[int, int]) -> None:
        """
        Set snake direction, preventing immediate reversals.
        
        Args:
            dir_tuple: Direction vector (dx, dy)
        """
        dx, dy = dir_tuple
        # Prevent snake from reversing into itself
        if (dx, dy) == (-self.direction[0], -self.direction[1]):
            return
        self.direction = dir_tuple

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


class Fruit:
    """Fruit entity that the snake can eat to grow."""
    
    def __init__(self, grid_size: int, snake_body: list):
        """
        Initialize fruit at random valid position.
        
        Args:
            grid_size: Size of the game grid
            snake_body: Current snake body positions to avoid
        """
        self.grid_size = grid_size
        self.position = self._new_position(snake_body)

    def _new_position(self, snake_body: list) -> Tuple[int, int]:
        """Generate new fruit position avoiding snake body."""
        while True:
            pos = random_position(self.grid_size)
            if pos not in snake_body:
                return pos

    def respawn(self, snake_body: list) -> None:
        """Respawn fruit at new valid position."""
        self.position = self._new_position(snake_body)


class SnakeGame:
    """
    Main Snake game implementation with both human and AI modes.
    Handles game logic, rendering, and state management.
    """
    
    def __init__(
        self, 
        grid_size: int = 12, 
        cell_size: int = 32, 
        mode: str = "human",
        reward_fruit: float = 5.0,
        reward_death: float = -10.0,
        reward_step: float = -0.01,
        reward_win: float = 100.0
    ):
        """
        Initialize the Snake game.
        
        Args:
            grid_size: Size of square game grid
            cell_size: Size of each grid cell in pixels
            mode: Game mode ('human' or 'ai')
            reward_fruit: Reward for eating fruit
            reward_death: Penalty for dying
            reward_step: Small penalty per step (encourages efficiency)
            reward_win: Bonus for achieving perfect game
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.mode = mode
        
        # Game entities
        self.snake = Snake(grid_size)
        self.fruit = Fruit(grid_size, self.snake.body)
        
        # Game state
        self.score = 0
        self.running = True
        self.won = False
        
        # Reward system for RL training
        self.reward_fruit = reward_fruit
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_win = reward_win
        
        logger.debug(f"SnakeGame initialized: {grid_size}x{grid_size}, mode={mode}")

    def check_win_condition(self) -> bool:
        """
        Check if snake has achieved perfect game (filled entire grid).
        
        Returns:
            True if snake has won, False otherwise
        """
        current_length = len(self.snake.body)
        if self.snake.grow:
            current_length += 1
        return current_length == self.grid_size * self.grid_size

    def update(self) -> None:
        """Update game state for one step."""
        self.snake.move()
        
        # Check fruit collection
        if self.snake.head() == self.fruit.position:
            self.snake.grow_snake()
            self.score += 1
            
            # Check win condition after eating fruit
            if self.check_win_condition():
                self.won = True
                self.running = False
                logger.info(f"Perfect game achieved! Score: {self.score}")
                return
            
            # Respawn fruit if game continues
            self.fruit.respawn(self.snake.body)
            
        # Check collision conditions
        if self.snake.collided_with_self() or self.snake.collided_with_wall():
            self.running = False
            logger.debug(f"Game over. Final score: {self.score}")

    def ai_step(self, action_idx: int, device: torch.device) -> None:
        """
        Execute one AI step with given action.
        
        Args:
            action_idx: Index of action to take
            device: PyTorch device for tensor operations
        """
        self.step(action_idx, device)

    def step(self, action_idx: int, device: torch.device) -> Tuple[torch.Tensor, float, bool]:
        """
        Enhanced step function with smarter reward shaping.
        
        Args:
            action_idx: Index of action to take
            device: PyTorch device for tensor operations
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        from agent.dqn import ACTIONS
        
        if not isinstance(action_idx, int) or action_idx not in range(len(ACTIONS)):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        # Store previous state for advanced rewards
        prev_head = self.snake.head()
        prev_fruit_distance = abs(prev_head[0] - self.fruit.position[0]) + abs(prev_head[1] - self.fruit.position[1])
        
        # Apply action
        self.snake.set_direction(ACTIONS[action_idx])
        prev_score = self.score
        self.update()
        
        # Enhanced reward calculation
        if not self.running:
            if self.won:
                reward = 100.0  # Perfect game bonus
            else:
                reward = -15.0  # Death penalty (increased)
            done = True
        elif self.score > prev_score:
            # Ate fruit - big reward
            reward = 20.0  # Increased fruit reward
            done = False
        else:
            # Advanced reward shaping for smarter behavior
            current_head = self.snake.head()
            current_fruit_distance = abs(current_head[0] - self.fruit.position[0]) + abs(current_head[1] - self.fruit.position[1])
            
            # Reward getting closer to fruit
            if current_fruit_distance < prev_fruit_distance:
                reward = 0.5  # Small reward for approaching fruit
            else:
                reward = -0.1  # Small penalty for moving away
            
            # Additional penalty for getting too close to walls or self
            danger_penalty = 0.0
            head_x, head_y = current_head
            
            # Near wall penalty
            if head_x <= 1 or head_x >= self.grid_size-2 or head_y <= 1 or head_y >= self.grid_size-2:
                danger_penalty -= 0.5
            
            # Near self penalty
            if current_head in self.snake.body[1:4]:  # Check next 3 body segments
                danger_penalty -= 1.0
            
            reward += danger_penalty
            done = False
        
        next_state = self.get_state(device)
        return next_state, reward, done

    def get_state(self, device: torch.device) -> torch.Tensor:
        """
        Get current game state as tensor for neural network input.
        
        Args:
            device: PyTorch device for tensor placement
            
        Returns:
            State tensor with shape (grid_size*grid_size + 6,)
        """
        # Grid encoding (snake=1.0, fruit=2.0, empty=0.0)
        state = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32, device=device)
        
        # Encode snake body
        for (x, y) in self.snake.body:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                state[x, y] = 1.0
        
        # Encode fruit
        fx, fy = self.fruit.position
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            state[fx, fy] = 2.0

        # Direction one-hot encoding (UP, DOWN, LEFT, RIGHT)
        dir_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        direction = torch.zeros(4, device=device)
        direction[dir_map.get(self.snake.direction, 0)] = 1.0

        # Relative fruit position (normalized to [-1, 1])
        head_x, head_y = self.snake.head()
        dx = (fx - head_x) / (self.grid_size - 1)
        dy = (fy - head_y) / (self.grid_size - 1)
        rel_fruit = torch.tensor([dx, dy], dtype=torch.float32, device=device)

        # Concatenate all features
        flat_grid = state.flatten()
        full_state = torch.cat([flat_grid, direction, rel_fruit])
        
        return full_state

    def draw(self, screen: pygame.Surface, board_offset_x: int = 0, board_offset_y: int = 0) -> None:
        """
        Render the game to the screen.
        
        Args:
            screen: Pygame surface to draw on
            board_offset_x: Horizontal offset for game board
            board_offset_y: Vertical offset for game board
        """
        # Modern dark gradient background
        for y in range(screen.get_height()):
            color = (
                26 + int(20 * y / screen.get_height()),
                26 + int(20 * y / screen.get_height()),
                32 + int(40 * y / screen.get_height())
            )
            pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))

        # Draw grid border
        grid_w = self.grid_size * self.cell_size
        grid_h = self.grid_size * self.cell_size
        border_rect = pygame.Rect(board_offset_x, board_offset_y, grid_w, grid_h)
        pygame.draw.rect(screen, (80, 80, 100), border_rect, width=4, border_radius=12)

        # Draw snake with rounded segments
        for segment in self.snake.body:
            pygame.draw.rect(
                screen, (0, 180, 45),  # Bright green
                (
                    board_offset_x + segment[0] * self.cell_size,
                    board_offset_y + segment[1] * self.cell_size,
                    self.cell_size, self.cell_size
                ),
                border_radius=7
            )
        
        # Draw fruit (only if game not won)
        if not self.won:
            fx, fy = self.fruit.position
            pygame.draw.ellipse(
                screen, (220, 60, 60),  # Bright red
                (
                    board_offset_x + fx * self.cell_size,
                    board_offset_y + fy * self.cell_size,
                    self.cell_size, self.cell_size
                )
            )
        
        # Draw scoreboard
        self.draw_scoreboard(screen, board_offset_x, board_offset_y)

    def draw_game_over(self, screen: pygame.Surface) -> None:
        """Draw game over message."""
        font_size = max(20, int(min(screen.get_width(), screen.get_height()) // 10))
        
        if self.won:
            message = f"🏆 PERFECT GAME! Score: {self.score} (R to Restart)"
            color = (0, 255, 0)  # Green for victory
        else:
            message = f"Game Over! Score: {self.score} (R to Restart)"
            color = (255, 255, 255)  # White for game over
            
        font = pygame.font.SysFont("arial", font_size)
        text_surface = font.render(message, True, color)
        text_rect = text_surface.get_rect(center=screen.get_rect().center)

        # Adjust font size if text too wide
        while text_rect.width > screen.get_width() * 0.95 and font_size > 10:
            font_size -= 2
            font = pygame.font.SysFont("arial", font_size)
            text_surface = font.render(message, True, color)
            text_rect = text_surface.get_rect(center=screen.get_rect().center)

        screen.blit(text_surface, text_rect)
    
    def draw_scoreboard(self, screen: pygame.Surface, board_offset_x: int = 0, board_offset_y: int = 0) -> None:
        """Draw score display above the game board."""
        font = pygame.font.SysFont("arial", 24)
        score_text = f"Score: {self.score}"
        if self.won:
            score_text += " - PERFECT! 🏆"
        
        text_surface = font.render(score_text, True, (255, 255, 255))
        screen.blit(text_surface, (board_offset_x, board_offset_y - 35))

    def reset(self) -> None:
        """Reset game to initial state."""
        self.__init__(
            self.grid_size, 
            self.cell_size, 
            self.mode,
            reward_fruit=self.reward_fruit,
            reward_death=self.reward_death,
            reward_step=self.reward_step,
            reward_win=self.reward_win
        )
        logger.debug("SnakeGame reset")

