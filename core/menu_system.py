"""
Menu System for SlytherNN - Professional UI and menu management.
Handles main menu, settings, game over screens, and user interactions.
"""

import pygame
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import logging

from config import GameConfig, UIConfig

logger = logging.getLogger(__name__)


class MenuResult(Enum):
    """Enumeration of possible menu results."""
    AI_MODE = "ai"
    HUMAN_MODE = "human"
    SETTINGS = "settings"
    QUIT = "quit"
    BACK = "back"


class MenuSystem:
    """
    Professional menu system with multiple screens and smooth interactions.
    Handles main menu, settings, and various UI overlays.
    """
    
    def __init__(self, game_config: GameConfig, ui_config: UIConfig):
        """
        Initialize the menu system.
        
        Args:
            game_config: Game configuration settings
            ui_config: UI configuration settings
        """
        self.game_config = game_config
        self.ui_config = ui_config
        
        # Menu state
        self.current_menu = "main"
        self.selected_option = 0
        self.menu_transition_alpha = 255
        
        # Fonts (will be initialized when first used)
        self._fonts: Dict[str, pygame.font.Font] = {}
        
        # Menu options for different screens
        self.main_menu_options = [
            {"text": "Play with AI", "action": MenuResult.AI_MODE, "key_hint": "[SPACE]"},
            {"text": "Play as Human", "action": MenuResult.HUMAN_MODE, "key_hint": "[ARROW KEYS]"},
            {"text": "Settings", "action": MenuResult.SETTINGS, "key_hint": "[S]"},
            {"text": "Quit", "action": MenuResult.QUIT, "key_hint": "[ESC]"}
        ]
        
        logger.info("MenuSystem initialized")
    
    def _get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Get cached font or create new one."""
        key = f"{size}_{bold}"
        if key not in self._fonts:
            try:
                if bold:
                    self._fonts[key] = pygame.font.SysFont(self.game_config.font_family, size, bold=True)
                else:
                    self._fonts[key] = pygame.font.SysFont(self.game_config.font_family, size)
            except Exception:
                # Fallback to default font
                self._fonts[key] = pygame.font.Font(None, size)
        return self._fonts[key]
    
    def show_main_menu(self, screen: pygame.Surface) -> Optional[str]:
        """
        Display the main menu and handle user input.
        
        Args:
            screen: Pygame surface to draw on
            
        Returns:
            Selected mode string or None to quit
        """
        clock = pygame.time.Clock()
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                result = self._handle_main_menu_event(event)
                if result is not None:
                    if result == MenuResult.QUIT:
                        return None
                    elif result == MenuResult.SETTINGS:
                        self._show_settings_menu(screen)
                        continue
                    else:
                        return result.value
            
            # Render menu
            self._render_main_menu(screen)
            pygame.display.flip()
            clock.tick(60)
    
    def _handle_main_menu_event(self, event: pygame.event.Event) -> Optional[MenuResult]:
        """Handle input events for main menu."""
        if event.type == pygame.KEYDOWN:
            # Direct key shortcuts
            if event.key == pygame.K_ESCAPE:
                return MenuResult.QUIT
            elif event.key == pygame.K_SPACE:
                return MenuResult.AI_MODE
            elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                return MenuResult.HUMAN_MODE
            elif event.key == pygame.K_s:
                return MenuResult.SETTINGS
            
            # Menu navigation
            elif event.key == pygame.K_w or event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.main_menu_options)
            elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.main_menu_options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                return self.main_menu_options[self.selected_option]["action"]
        
        elif event.type == pygame.MOUSEMOTION:
            # Mouse hover selection
            mouse_pos = pygame.mouse.get_pos()
            menu_y_start = self.game_config.screen_height // 2 - 50
            
            for i, option in enumerate(self.main_menu_options):
                option_y = menu_y_start + i * 60
                if option_y <= mouse_pos[1] <= option_y + 40:
                    self.selected_option = i
                    break
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                return self.main_menu_options[self.selected_option]["action"]
        
        return None
    
    def _render_main_menu(self, screen: pygame.Surface) -> None:
        """Render the main menu screen."""
        # Clear screen with gradient background
        self._draw_gradient_background(screen)
        
        # Draw title
        self._draw_title(screen)
        
        # Draw menu options
        self._draw_menu_options(screen)
        
        # Draw footer info
        self._draw_footer_info(screen)
        
        # Draw version info
        self._draw_version_info(screen)
    
    def _draw_gradient_background(self, screen: pygame.Surface) -> None:
        """Draw a subtle gradient background."""
        base_color = self.game_config.background_color
        for y in range(self.game_config.screen_height):
            # Create subtle gradient effect
            intensity = int(255 * (1.0 - y / self.game_config.screen_height * 0.3))
            color = tuple(min(255, max(0, c + intensity // 10)) for c in base_color)
            pygame.draw.line(screen, color, (0, y), (self.game_config.screen_width, y))
    
    def _draw_title(self, screen: pygame.Surface) -> None:
        """Draw the game title with style."""
        title_font = self._get_font(48, bold=True)
        subtitle_font = self._get_font(20)
        
        # Main title
        title_text = title_font.render("SlytherNN", True, (0, 255, 100))  # Bright green
        title_rect = title_text.get_rect(center=(self.game_config.screen_width // 2, 120))
        
        # Draw title with subtle shadow
        shadow_text = title_font.render("SlytherNN", True, (0, 100, 50))
        shadow_rect = shadow_text.get_rect(center=(title_rect.centerx + 2, title_rect.centery + 2))
        screen.blit(shadow_text, shadow_rect)
        screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = subtitle_font.render("Deep Q-Network Snake AI", True, self.game_config.text_color)
        subtitle_rect = subtitle_text.get_rect(center=(self.game_config.screen_width // 2, 160))
        screen.blit(subtitle_text, subtitle_rect)
    
    def _draw_menu_options(self, screen: pygame.Surface) -> None:
        """Draw menu options with selection highlighting."""
        option_font = self._get_font(24)
        hint_font = self._get_font(16)
        
        menu_y_start = self.game_config.screen_height // 2 - 50
        
        for i, option in enumerate(self.main_menu_options):
            y_pos = menu_y_start + i * 60
            is_selected = (i == self.selected_option)
            
            # Selection background
            if is_selected:
                selection_rect = pygame.Rect(
                    self.game_config.screen_width // 2 - 150, 
                    y_pos - 5, 
                    300, 40
                )
                pygame.draw.rect(screen, (40, 40, 50), selection_rect, border_radius=5)
                pygame.draw.rect(screen, (0, 255, 100), selection_rect, width=2, border_radius=5)
            
            # Option text
            text_color = (255, 255, 255) if is_selected else (200, 200, 200)
            option_text = option_font.render(option["text"], True, text_color)
            text_rect = option_text.get_rect(center=(self.game_config.screen_width // 2, y_pos + 15))
            screen.blit(option_text, text_rect)
            
            # Key hint
            hint_text = hint_font.render(option["key_hint"], True, (150, 150, 150))
            hint_rect = hint_text.get_rect(center=(self.game_config.screen_width // 2, y_pos + 35))
            screen.blit(hint_text, hint_rect)
    
    def _draw_footer_info(self, screen: pygame.Surface) -> None:
        """Draw footer information and controls."""
        info_font = self._get_font(14)
        
        footer_y = self.game_config.screen_height - 80
        
        controls = [
            "Navigation: W/S or ↑/↓",
            "Select: ENTER or Mouse Click",
            "Quick Start: SPACE (AI) or Arrow Keys (Human)"
        ]
        
        for i, control in enumerate(controls):
            text = info_font.render(control, True, (150, 150, 150))
            text_rect = text.get_rect(center=(self.game_config.screen_width // 2, footer_y + i * 20))
            screen.blit(text, text_rect)
    
    def _draw_version_info(self, screen: pygame.Surface) -> None:
        """Draw version and build information."""
        version_font = self._get_font(12)
        
        version_text = version_font.render("v2.0.0 - Professional Edition", True, (100, 100, 100))
        version_rect = version_text.get_rect(bottomright=(self.game_config.screen_width - 10, self.game_config.screen_height - 10))
        screen.blit(version_text, version_rect)
    
    def _show_settings_menu(self, screen: pygame.Surface) -> None:
        """Show settings menu (placeholder for now)."""
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE:
                        return
            
            # Draw settings screen
            screen.fill(self.game_config.background_color)
            
            title_font = self._get_font(36, bold=True)
            title_text = title_font.render("Settings", True, self.game_config.text_color)
            title_rect = title_text.get_rect(center=(self.game_config.screen_width // 2, 100))
            screen.blit(title_text, title_rect)
            
            # Placeholder settings
            info_font = self._get_font(20)
            settings_info = [
                "Settings menu coming soon!",
                "",
                "Future features:",
                "• Audio settings",
                "• Graphics options", 
                "• AI difficulty levels",
                "• Custom key bindings",
                "",
                "Press ESC to return"
            ]
            
            for i, line in enumerate(settings_info):
                color = self.game_config.text_color if line else (0, 0, 0)
                text = info_font.render(line, True, color)
                text_rect = text.get_rect(center=(self.game_config.screen_width // 2, 200 + i * 30))
                screen.blit(text, text_rect)
            
            pygame.display.flip()
            clock.tick(60)
    
    def show_game_over_overlay(self, screen: pygame.Surface, score: int, ai_mode: bool = False) -> bool:
        """
        Show game over overlay with score and restart option.
        
        Args:
            screen: Pygame surface to draw on
            score: Final score achieved
            ai_mode: Whether this was an AI game
            
        Returns:
            True to restart, False to return to menu
        """
        overlay = pygame.Surface((self.game_config.screen_width, self.game_config.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Game Over title
        title_font = self._get_font(48, bold=True)
        title_text = title_font.render("Game Over", True, (255, 100, 100))
        title_rect = title_text.get_rect(center=(self.game_config.screen_width // 2, 200))
        screen.blit(title_text, title_rect)
        
        # Score
        score_font = self._get_font(24)
        mode_text = "AI Score" if ai_mode else "Your Score"
        score_text = score_font.render(f"{mode_text}: {score}", True, self.game_config.text_color)
        score_rect = score_text.get_rect(center=(self.game_config.screen_width // 2, 260))
        screen.blit(score_text, score_rect)
        
        # Instructions
        instruction_font = self._get_font(20)
        instructions = [
            "Press R to Restart",
            "Press ESC to Return to Menu",
            "Press SPACE for AI Analysis" if ai_mode else ""
        ]
        
        for i, instruction in enumerate(instructions):
            if instruction:  # Skip empty strings
                text = instruction_font.render(instruction, True, (200, 200, 200))
                text_rect = text.get_rect(center=(self.game_config.screen_width // 2, 320 + i * 30))
                screen.blit(text, text_rect)
        
        return False  # This is just rendering, actual input handling is in GameController
    
    def show_loading_screen(self, screen: pygame.Surface, message: str = "Loading...") -> None:
        """Show loading screen with message."""
        screen.fill(self.game_config.background_color)
        
        # Loading message
        font = self._get_font(24)
        text = font.render(message, True, self.game_config.text_color)
        text_rect = text.get_rect(center=(self.game_config.screen_width // 2, self.game_config.screen_height // 2))
        screen.blit(text, text_rect)
        
        # Simple loading animation (dots)
        import time
        dots = "." * (int(time.time() * 2) % 4)
        dots_text = font.render(dots, True, self.game_config.text_color)
        dots_rect = dots_text.get_rect(center=(self.game_config.screen_width // 2, self.game_config.screen_height // 2 + 40))
        screen.blit(dots_text, dots_rect)
        
        pygame.display.flip()
    
    def cleanup(self) -> None:
        """Clean up menu system resources."""
        self._fonts.clear()
        logger.info("MenuSystem cleanup complete")

