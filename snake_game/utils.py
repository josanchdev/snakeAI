"""
Utility functions for Snake game mechanics.
"""

import random
from typing import Tuple


def random_position(grid_size: int) -> Tuple[int, int]:
    """
    Generate random position within grid boundaries.
    
    Args:
        grid_size: Size of the square grid
        
    Returns:
        Tuple of (x, y) coordinates
    """
    return (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_valid_position(pos: Tuple[int, int], grid_size: int) -> bool:
    """
    Check if position is within grid boundaries.
    
    Args:
        pos: Position to check (x, y)
        grid_size: Size of the square grid
        
    Returns:
        True if position is valid, False otherwise
    """
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size

