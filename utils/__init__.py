"""
Utility modules for SlytherNN project.
Provides device management, file operations, and common helper functions.
"""

from .device_utils import (
    DeviceManager,
    get_device_manager,
    get_device,
    setup_device_for_training,
    get_optimal_batch_size,
    get_memory_info,
    clear_device_memory
)

from .file_utils import (
    FileManager,
    get_file_manager,
    ensure_project_directories,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
    safe_save_json,
    safe_load_json
)

__all__ = [
    # Device utilities
    "DeviceManager",
    "get_device_manager", 
    "get_device",
    "setup_device_for_training",
    "get_optimal_batch_size",
    "get_memory_info",
    "clear_device_memory",
    
    # File utilities
    "FileManager",
    "get_file_manager",
    "ensure_project_directories", 
    "get_latest_checkpoint",
    "cleanup_old_checkpoints",
    "safe_save_json",
    "safe_load_json"
]

