"""
File utilities for path management, file operations, and data persistence.
Handles checkpoint loading, directory management, and file validation.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations and path handling for the project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """Create necessary project directories if they don't exist."""
        directories = [
            "checkpoints",
            "logs", 
            "plots",
            "results",
            "analysis"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def get_checkpoint_files(self, pattern: str = "*.pth") -> List[Path]:
        """Get list of checkpoint files sorted by modification time."""
        checkpoint_dir = self.project_root / "checkpoints"
        files = list(checkpoint_dir.glob(pattern))
        return sorted(files, key=lambda x: x.stat().st_mtime)
    
    def get_latest_checkpoint(self, pattern: str = "dqn_snake_checkpoint_ep*.pth") -> Optional[Path]:
        """Get the most recent checkpoint file."""
        files = self.get_checkpoint_files(pattern)
        if not files:
            return None
            
        # Sort by episode number if available
        try:
            files.sort(key=lambda x: int(x.stem.split('_ep')[1]))
            return files[-1]
        except (IndexError, ValueError):
            # Fallback to modification time
            return max(files, key=lambda x: x.stat().st_mtime)
    
    def cleanup_old_checkpoints(self, keep_last: int = 3, pattern: str = "dqn_snake_checkpoint_ep*.pth") -> int:
        """Remove old checkpoint files, keeping only the most recent ones."""
        files = self.get_checkpoint_files(pattern)
        
        if len(files) <= keep_last:
            return 0
            
        files_to_remove = files[:-keep_last]
        removed_count = 0
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
                logger.info(f"Removed old checkpoint: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        
        return removed_count
    
    def safe_save_json(self, data: Dict[Any, Any], filepath: Union[str, Path]) -> bool:
        """Safely save data to JSON with atomic write."""
        filepath = Path(filepath)
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            # Write to temporary file first
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            temp_filepath.replace(filepath)
            logger.debug(f"Saved JSON to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            return False
    
    def safe_load_json(self, filepath: Union[str, Path]) -> Optional[Dict[Any, Any]]:
        """Safely load JSON data with error handling."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"JSON file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON from: {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None
    
    def backup_file(self, filepath: Union[str, Path], backup_dir: Optional[str] = None) -> Optional[Path]:
        """Create a timestamped backup of a file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Cannot backup non-existent file: {filepath}")
            return None
        
        # Determine backup directory
        if backup_dir:
            backup_path = Path(backup_dir)
        else:
            backup_path = filepath.parent / "backups"
        
        backup_path.mkdir(exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_filepath = backup_path / backup_filename
        
        try:
            shutil.copy2(filepath, backup_filepath)
            logger.info(f"Created backup: {backup_filepath}")
            return backup_filepath
            
        except Exception as e:
            logger.error(f"Failed to create backup of {filepath}: {e}")
            return None
    
    def get_file_size_mb(self, filepath: Union[str, Path]) -> float:
        """Get file size in megabytes."""
        filepath = Path(filepath)
        if filepath.exists():
            return filepath.stat().st_size / (1024 * 1024)
        return 0.0
    
    def get_directory_size_mb(self, directory: Union[str, Path]) -> float:
        """Get total size of directory in megabytes."""
        directory = Path(directory)
        if not directory.exists():
            return 0.0
        
        total_size = 0
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                total_size += filepath.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def validate_checkpoint_file(self, filepath: Union[str, Path]) -> Tuple[bool, str]:
        """Validate that a checkpoint file is readable and contains expected keys."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return False, "File does not exist"
        
        if filepath.suffix != '.pth':
            return False, "Not a .pth file"
        
        try:
            import torch
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Check for expected keys
            if isinstance(checkpoint, dict):
                required_keys = ['model']  # Minimum required
                missing_keys = [key for key in required_keys if key not in checkpoint]
                if missing_keys:
                    return False, f"Missing keys: {missing_keys}"
            
            return True, "Valid checkpoint file"
            
        except Exception as e:
            return False, f"Error loading checkpoint: {str(e)}"


# Global file manager instance
_file_manager = None


def get_file_manager() -> FileManager:
    """Get the global file manager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


def ensure_project_directories() -> None:
    """Ensure all project directories exist."""
    get_file_manager().ensure_directories()


def get_latest_checkpoint(pattern: str = "dqn_snake_checkpoint_ep*.pth") -> Optional[Path]:
    """Get the most recent checkpoint file."""
    return get_file_manager().get_latest_checkpoint(pattern)


def cleanup_old_checkpoints(keep_last: int = 3) -> int:
    """Remove old checkpoint files."""
    return get_file_manager().cleanup_old_checkpoints(keep_last)


def safe_save_json(data: Dict[Any, Any], filepath: Union[str, Path]) -> bool:
    """Safely save data to JSON file."""
    return get_file_manager().safe_save_json(data, filepath)


def safe_load_json(filepath: Union[str, Path]) -> Optional[Dict[Any, Any]]:
    """Safely load JSON data."""
    return get_file_manager().safe_load_json(filepath)

