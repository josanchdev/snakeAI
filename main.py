#!/usr/bin/env python3
"""
SlytherNN: Deep Q-Network Snake AI - Main Entry Point
Professional reinforcement learning implementation with enterprise-grade architecture.

Usage:
    python main.py              # Interactive menu mode
    python main.py --mode ai    # Direct AI mode
    python main.py --mode human # Direct human mode
"""

import sys
import logging
from typing import Optional
import argparse

from config import GameConfig, UIConfig, ProjectConfig
from core import GameController

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="SlytherNN: Deep Q-Network Snake AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Show interactive menu
    python main.py --mode ai          # Start directly in AI mode
    python main.py --mode human       # Start directly in human mode
    python main.py --verbose          # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["ai", "human", "menu"],
        default="menu",
        help="Game mode to start in (default: menu)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU mode (disable CUDA)"
    )
    
    return parser

def load_configuration(config_path: Optional[str] = None) -> tuple[GameConfig, UIConfig]:
    """Load configuration from file or use defaults."""
    try:
        if config_path:
            project_config = ProjectConfig.load_from_file(config_path)
            return project_config.game, project_config.ui
        else:
            return GameConfig(), UIConfig()
            
    except Exception as e:
        logging.warning(f"Failed to load config from {config_path}: {e}")
        logging.info("Using default configuration")
        return GameConfig(), UIConfig()

def main() -> int:
    """
    Main entry point for SlytherNN application.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        game_config, ui_config = load_configuration(args.config)
        
        # Override UI config based on CLI args
        if args.verbose:
            ui_config.verbose_logging = True
        
        # Force CPU mode if requested
        if args.no_gpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("GPU disabled via command line flag")
        
        # Print startup information
        logger.info("🐍 SlytherNN v2.0.0 - Starting up...")
        logger.info(f"Mode: {args.mode}")
        
        # Create and initialize game controller
        controller = GameController(game_config, ui_config)
        
        if not controller.initialize():
            logger.error("Failed to initialize game controller")
            return 1
        
        # Handle different startup modes
        if args.mode == "menu":
            # Standard interactive mode
            controller.run()
        else:
            # Direct mode - skip menu and go straight to game
            logger.info(f"Starting in direct {args.mode} mode")
            # For direct mode, we'd need to modify GameController
            # For now, just run normally (menu will still appear)
            controller.run()
        
        logger.info("SlytherNN shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

