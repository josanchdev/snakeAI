"""
Game Controller for SlytherNN - Professional game orchestration and state management.
Handles game mode switching, AI integration, and user interface coordination.
"""

import pygame
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from config import GameConfig, UIConfig
from core.ai_manager import AIManager
from core.menu_system import MenuSystem
from utils import get_device, setup_device_for_training
from snake_game.game import SnakeGame

logger = logging.getLogger(__name__)


class GameController:
    """
    Main game controller that orchestrates all game components.
    Handles mode switching, AI integration, and game flow management.
    """
    
    def __init__(self, game_config: Optional[GameConfig] = None, ui_config: Optional[UIConfig] = None):
        """
        Initialize the game controller.
        
        Args:
            game_config: Game configuration settings
            ui_config: UI configuration settings  
        """
        self.game_config = game_config or GameConfig()
        self.ui_config = ui_config or UIConfig()
        
        # Core components
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.menu_system: Optional[MenuSystem] = None
        self.ai_manager: Optional[AIManager] = None
        self.game: Optional[SnakeGame] = None
        
        # Game state
        self.running = False
        self.current_mode: Optional[str] = None
        self.device = get_device()
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        
        logger.info(f"GameController initialized with device: {self.device}")
    
    def initialize(self) -> bool:
        """
        Initialize all game systems and components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize pygame
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.game_config.screen_width,
                self.game_config.screen_height
            ))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("SlytherNN: Snake RL")
            
            # Initialize subsystems
            self.menu_system = MenuSystem(self.game_config, self.ui_config)
            self.ai_manager = AIManager(self.game_config.grid_size)
            
            # Set up device for optimal performance
            if self.ui_config.verbose_logging:
                setup_device_for_training()
            
            logger.info("✅ GameController initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GameController: {e}")
            return False
    
    def run(self) -> None:
        """Main game loop - entry point for the application."""
        if not self.initialize():
            logger.error("Failed to initialize game controller")
            return
        
        try:
            self.running = True
            self._main_game_loop()
            
        except KeyboardInterrupt:
            logger.info("Game interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in game loop: {e}")
        finally:
            self.cleanup()
    
    def _main_game_loop(self) -> None:
        """Main application loop handling menu and game modes."""
        while self.running:
            # Show main menu and get mode selection
            selected_mode = self.menu_system.show_main_menu(self.screen)
            
            if selected_mode is None:
                # User chose to quit
                self.running = False
                continue
            
            # Set up and run the selected game mode
            if self._setup_game_mode(selected_mode):
                self._run_game_mode(selected_mode)
            else:
                logger.error(f"Failed to setup game mode: {selected_mode}")
    
    def _setup_game_mode(self, mode: str) -> bool:
        """
        Set up the game for the specified mode.
        
        Args:
            mode: Game mode ('ai' or 'human')
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.current_mode = mode
            
            # Initialize game instance
            self.game = SnakeGame(
                self.game_config.grid_size,
                self.game_config.cell_size,
                mode=mode
            )
            
            # Load AI model if needed
            if mode == "ai":
                if not self._setup_ai_mode():
                    # Fallback to human mode
                    logger.warning("AI setup failed, falling back to human mode")
                    self.current_mode = "human"
                    self.game.mode = "human"
            
            logger.info(f"Game mode setup complete: {self.current_mode}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup game mode {mode}: {e}")
            return False
    
    def _setup_ai_mode(self) -> bool:
        """
        Set up AI mode by loading the trained model.
        
        Returns:
            True if AI model loaded successfully, False otherwise
        """
        try:
            success = self.ai_manager.load_model()
            if success:
                model_info = self.ai_manager.get_model_info()
                logger.info(f"AI model loaded: Episode {model_info.get('episode', 'unknown')}")
                
                if self.ui_config.verbose_logging:
                    self._log_ai_model_info(model_info)
                
                return True
            else:
                logger.error("No trained AI model found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup AI mode: {e}")
            return False
    
    def _log_ai_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log detailed AI model information."""
        print(f"\n🤖 AI Model Information:")
        print(f"   Episode: {model_info.get('episode', 'unknown')}")
        print(f"   Parameters: {model_info.get('parameters', 'unknown'):,}")
        print(f"   Device: {model_info.get('device', 'unknown')}")
        print(f"   Input Dimension: {model_info.get('input_dim', 'unknown')}")
        print(f"   Actions: {len(model_info.get('actions', []))}")
    
    def _run_game_mode(self, mode: str) -> None:
        """
        Run the game in the specified mode.
        
        Args:
            mode: Game mode to run
        """
        if not self.game or not self.screen:
            logger.error("Game or screen not initialized")
            return
        
        # Set up game timer
        move_event = pygame.USEREVENT + 1
        pygame.time.set_timer(move_event, self.game_config.ai_move_delay_ms)
        
        game_running = True
        
        while game_running and self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    game_running = False
                    
                elif event.type == pygame.KEYDOWN:
                    game_running = self._handle_keydown(event, mode)
                    
                elif event.type == move_event and self.game.running:
                    self._handle_game_step(mode)
            
            # Render frame
            self._render_frame()
            
            # Update performance counters
            self._update_performance_counters()
            
            # Control frame rate
            self.clock.tick(self.game_config.fps)
        
        # Clean up timer
        pygame.time.set_timer(move_event, 0)
    
    def _handle_keydown(self, event: pygame.event.Event, mode: str) -> bool:
        """
        Handle keyboard input.
        
        Args:
            event: Pygame keyboard event
            mode: Current game mode
            
        Returns:
            False to exit game mode, True to continue
        """
        if event.key == pygame.K_ESCAPE:
            return False  # Exit to main menu
        
        # Human mode controls
        if mode == "human" and self.game:
            direction_map = {
                pygame.K_UP: (0, -1),
                pygame.K_DOWN: (0, 1), 
                pygame.K_LEFT: (-1, 0),
                pygame.K_RIGHT: (1, 0)
            }
            
            if event.key in direction_map:
                self.game.snake.set_direction(direction_map[event.key])
        
        # Restart game
        if event.key == pygame.K_r and self.game and not self.game.running:
            self.game.reset()
            logger.info("Game reset by user")
        
        # Show AI analysis (if in AI mode)
        if event.key == pygame.K_SPACE and mode == "ai" and self.ai_manager and self.game:
            self._show_ai_analysis()
        
        return True
    
    def _handle_game_step(self, mode: str) -> None:
        """
        Handle one game step based on the current mode.
        
        Args:
            mode: Current game mode
        """
        if not self.game:
            return
        
        if mode == "ai" and self.ai_manager and self.ai_manager.is_loaded:
            try:
                # Get game state and predict action
                state = self.game.get_state(self.device).flatten()
                action_idx = self.ai_manager.predict_action(state)
                
                # Execute AI action
                self.game.ai_step(action_idx, self.device)
                
            except Exception as e:
                logger.error(f"AI step failed: {e}")
                # Fallback to random action or pause
                self.game.update()
        else:
            # Human mode or AI not available
            self.game.update()
    
    def _show_ai_analysis(self) -> None:
        """Show detailed AI analysis of current game state."""
        if not self.game or not self.ai_manager:
            return
        
        try:
            state = self.game.get_state(self.device).flatten()
            analysis = self.ai_manager.analyze_state(state)
            
            print(f"\n🧠 AI Analysis:")
            print(f"   Recommended Action: {analysis['recommended_action']['name']}")
            print(f"   Confidence: {analysis['confidence']:.2%}")
            print(f"   Q-Values: {analysis['q_values']}")
            
        except Exception as e:
            logger.error(f"Failed to show AI analysis: {e}")
    
    def _render_frame(self) -> None:
        """Render one frame of the game."""
        if not self.game or not self.screen:
            return
        
        # Clear screen
        self.screen.fill(self.game_config.background_color)
        
        # Draw game
        self.game.draw(
            self.screen,
            self.game_config.board_offset_x,
            self.game_config.board_offset_y
        )
        
        # Draw UI overlays
        if self.ui_config.show_fps:
            self._draw_fps_counter()
        
        if self.ui_config.show_score and self.game:
            self._draw_score()
        
        if self.ui_config.show_ai_info and self.current_mode == "ai":
            self._draw_ai_info()
        
        # Show game over screen
        if self.game and not self.game.running:
            self.game.draw_game_over(self.screen)
        
        pygame.display.flip()
    
    def _draw_fps_counter(self) -> None:
        """Draw FPS counter on screen."""
        if self.clock:
            fps = int(self.clock.get_fps())
            font = pygame.font.SysFont("arial", 20)
            fps_text = font.render(f"FPS: {fps}", True, self.game_config.text_color)
            self.screen.blit(fps_text, (10, 10))
    
    def _draw_score(self) -> None:
        """Draw current score on screen."""
        if self.game:
            score = len(self.game.snake.body) - 1  # Assuming this is how score is calculated
            font = pygame.font.SysFont("arial", 20)
            score_text = font.render(f"Score: {score}", True, self.game_config.text_color)
            self.screen.blit(score_text, (10, 35))
    
    def _draw_ai_info(self) -> None:
        """Draw AI information on screen."""
        if self.ai_manager and self.ai_manager.is_loaded:
            font = pygame.font.SysFont("arial", 16)
            info_text = font.render("AI Mode - Press SPACE for analysis", True, self.game_config.text_color)
            self.screen.blit(info_text, (10, self.game_config.screen_height - 25))
    
    def _update_performance_counters(self) -> None:
        """Update performance tracking counters."""
        self.frame_count += 1
        if self.clock:
            self.fps_counter = self.clock.get_fps()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            "frame_count": self.frame_count,
            "current_fps": self.fps_counter,
            "target_fps": self.game_config.fps,
            "mode": self.current_mode
        }
    
    def cleanup(self) -> None:
        """Clean up resources and shut down systems."""
        try:
            if self.ai_manager:
                self.ai_manager.unload_model()
            
            pygame.quit()
            logger.info("GameController cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the SlytherNN application."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run game controller
    controller = GameController()
    controller.run()


if __name__ == "__main__":
    main()

