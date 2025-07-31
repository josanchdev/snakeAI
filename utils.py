"""
Consolidated utilities for SlytherNN - Device management, file operations, and training helpers.
Combines device_utils.py and file_utils.py into a single, comprehensive module.
"""

import os
import json
import pickle
import shutil
import psutil
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Device Management (formerly device_utils.py)
# =============================================================================

class DeviceManager:
    """Enhanced device manager with automatic optimization detection."""
    
    def __init__(self):
        self.device = self._detect_optimal_device()
        self.device_info = self._get_device_info()
        self._apply_optimizations()
        
    def _detect_optimal_device(self) -> torch.device:
        """Detect and return the optimal PyTorch device."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                device = torch.device("cuda")
                
                # Log GPU details for high-end cards
                if torch.cuda.get_device_properties(0).total_memory > 20 * 1024**3:  # > 20GB
                    logger.info(f"🚀 High-end GPU detected: {torch.cuda.get_device_name()}")
                
                return device
            except Exception as e:
                logger.warning(f"CUDA available but not usable: {e}")
                return torch.device("cpu")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"MPS available but not usable: {e}")
                return torch.device("cpu")
        
        return torch.device("cpu")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            "device_type": self.device.type,
            "device_name": "Unknown",
            "memory_total": 0,
            "memory_available": 0,
            "compute_capability": None,
            "is_high_end": False
        }
        
        if self.device.type == "cuda":
            info.update(self._get_cuda_info())
        elif self.device.type == "mps":
            info.update(self._get_mps_info())
        else:
            info.update(self._get_cpu_info())
            
        return info
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        props = torch.cuda.get_device_properties(self.device)
        memory_gb = props.total_memory / (1024**3)
        
        # Enhanced high-end detection
        is_high_end = (
            memory_gb >= 16 or      # 16GB+ VRAM
            props.major >= 8 or     # Ampere architecture or newer
            "RTX" in props.name or  # RTX series
            "Tesla" in props.name   # Tesla series
        )
        
        return {
            "device_name": props.name,
            "memory_total": props.total_memory,
            "memory_total_gb": memory_gb,
            "memory_available": torch.cuda.memory_reserved(self.device),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "is_high_end": is_high_end,
            "architecture": self._get_gpu_architecture(props.major)
        }
    
    def _get_gpu_architecture(self, major: int) -> str:
        """Get GPU architecture name from compute capability."""
        arch_map = {
            6: "Pascal",
            7: "Volta/Turing", 
            8: "Ampere",
            9: "Hopper",
            10: "Blackwell"
        }
        return arch_map.get(major, f"Compute {major}.x")
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        return {
            "device_name": "CPU",
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "cpu_count": physical_cores,
            "cpu_count_logical": logical_cores,
            "is_high_end": physical_cores >= 8  # 8+ cores considered high-end
        }
    
    def _get_mps_info(self) -> Dict[str, Any]:
        """Get MPS (Apple Silicon) device information."""
        return {
            "device_name": "Apple Silicon GPU",
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "is_high_end": True  # Apple Silicon is generally high-performance
        }
    
    def _apply_optimizations(self) -> None:
        """Apply device-specific optimizations."""
        if self.device.type == "cuda" and self.device_info.get("is_high_end"):
            # High-end GPU optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("🔥 Applied high-end GPU optimizations")
    
    def get_optimal_batch_size(self, base_batch_size: int = 256) -> int:
        """Get optimal batch size based on hardware."""
        if self.device.type == "cuda":
            memory_gb = self.device_info.get("memory_total_gb", 8)
            
            # More aggressive scaling for modern GPUs
            if memory_gb >= 24:  # RTX 4090, A100, etc.
                return min(base_batch_size * 4, 1024)
            elif memory_gb >= 16:  # RTX 4070 Ti, RTX 3080, etc.
                return min(base_batch_size * 3, 768)
            elif memory_gb >= 12:  # RTX 4060 Ti, RTX 3060, etc.
                return min(base_batch_size * 2, 512)
            elif memory_gb >= 8:   # RTX 3050, GTX 1070, etc.
                return base_batch_size
        
        return base_batch_size // 2  # Conservative for CPU/MPS
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if self.device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
            total_gb = self.device_info.get("memory_total_gb", 0)
            
            return {
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "total_gb": total_gb,
                "utilization_percent": (reserved_gb / total_gb * 100) if total_gb > 0 else 0,
                "available_gb": max(0, total_gb - reserved_gb)
            }
        else:
            memory = psutil.virtual_memory()
            return {
                "used_gb": memory.used / 1024**3,
                "available_gb": memory.available / 1024**3,
                "total_gb": memory.total / 1024**3,
                "utilization_percent": memory.percent
            }
    
    def clear_memory(self) -> None:
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def print_device_info(self) -> None:
        """Print comprehensive device information."""
        print(f"\n🖥️  Hardware Configuration:")
        print(f"   Device: {self.device}")
        print(f"   Name: {self.device_info['device_name']}")
        
        if self.device.type == "cuda":
            memory_gb = self.device_info.get('memory_total_gb', 0)
            print(f"   VRAM: {memory_gb:.1f} GB")
            print(f"   Architecture: {self.device_info.get('architecture', 'Unknown')}")
            print(f"   Compute Capability: {self.device_info.get('compute_capability', 'Unknown')}")
            
            if self.device_info.get('is_high_end'):
                print(f"   🚀 High-End GPU Optimizations Enabled!")
                recommended_batch = self.get_optimal_batch_size()
                print(f"   💡 Recommended Batch Size: {recommended_batch}")
                
        elif self.device.type == "cpu":
            cores = self.device_info.get('cpu_count', 0)
            threads = self.device_info.get('cpu_count_logical', 0)
            print(f"   Cores: {cores} physical, {threads} logical")
            
            if self.device_info.get('is_high_end'):
                print(f"   🔥 High-End CPU Detected!")
        
        # Memory info
        memory_info = self.get_memory_usage()
        if self.device.type == "cuda":
            print(f"   Available VRAM: {memory_info['available_gb']:.1f} GB")
        else:
            print(f"   Available RAM: {memory_info['available_gb']:.1f} GB")


# Global device manager instance
_device_manager = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager

def get_device() -> torch.device:
    """Get the optimal PyTorch device."""
    return get_device_manager().device

def setup_device_for_training() -> torch.device:
    """Setup and optimize device for training."""
    manager = get_device_manager()
    manager.print_device_info()
    return manager.device

def get_optimal_batch_size(base_batch_size: int = 256) -> int:
    """Get hardware-optimized batch size."""
    return get_device_manager().get_optimal_batch_size(base_batch_size)

def clear_device_memory() -> None:
    """Clear device memory cache."""
    get_device_manager().clear_memory()


# =============================================================================
# File Management (formerly file_utils.py) 
# =============================================================================

class FileManager:
    """Enhanced file manager with comprehensive operations."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """Create necessary project directories."""
        directories = [
            "checkpoints",
            "logs", 
            "plots",
            "results",
            "models"  # Added models directory
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def get_checkpoint_files(self, pattern: str = "*.pth") -> List[Path]:
        """Get list of checkpoint files sorted by modification time."""
        checkpoint_dir = self.project_root / "checkpoints"
        files = list(checkpoint_dir.glob(pattern))
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_latest_checkpoint(self, pattern: str = "checkpoint_step_*.pth") -> Optional[Path]:
        """Get the most recent checkpoint file."""
        files = self.get_checkpoint_files(pattern)
        if not files:
            # Fallback to old naming convention
            files = self.get_checkpoint_files("dqn_snake_checkpoint_ep*.pth")
        
        if not files:
            return None
        
        # Try to sort by step/episode number
        try:
            if "step_" in pattern:
                files.sort(key=lambda x: int(x.stem.split('_step_')[1]), reverse=True)
            elif "_ep" in files[0].name:
                files.sort(key=lambda x: int(x.stem.split('_ep')[1]), reverse=True)
            return files[0]
        except (IndexError, ValueError):
            # Fallback to modification time
            return files[0]
    
    def cleanup_old_checkpoints(self, keep_last: int = 5, pattern: str = "checkpoint_step_*.pth") -> int:
        """Remove old checkpoint files, keeping only the most recent ones."""
        files = self.get_checkpoint_files(pattern)
        
        if len(files) <= keep_last:
            return 0
        
        files_to_remove = files[keep_last:]  # Already sorted by modification time
        removed_count = 0
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
                logger.debug(f"Removed old checkpoint: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        
        return removed_count
    
    def safe_save_json(self, data: Dict[Any, Any], filepath: Union[str, Path]) -> bool:
        """Safely save data to JSON with atomic write."""
        filepath = Path(filepath)
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
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
        
        backup_path.mkdir(exist_ok=True, parents=True)
        
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
                try:
                    total_size += filepath.stat().st_size
                except (OSError, PermissionError):
                    continue
        
        return total_size / (1024 * 1024)
    
    def validate_checkpoint_file(self, filepath: Union[str, Path]) -> Tuple[bool, str]:
        """Validate that a checkpoint file is readable and contains expected keys."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return False, "File does not exist"
        
        if filepath.suffix != '.pth':
            return False, "Not a .pth file"
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Check for expected keys
            if isinstance(checkpoint, dict):
                # Check for different checkpoint formats
                required_keys = []
                if 'online_net' in checkpoint:  # New format
                    required_keys = ['online_net']
                elif 'model' in checkpoint:     # Old format
                    required_keys = ['model']
                else:
                    return False, "No model state found in checkpoint"
                
                missing_keys = [key for key in required_keys if key not in checkpoint]
                if missing_keys:
                    return False, f"Missing keys: {missing_keys}"
            
            return True, "Valid checkpoint file"
            
        except Exception as e:
            return False, f"Error loading checkpoint: {str(e)}"
    
    def find_best_checkpoint(self, metric: str = "avg_score") -> Optional[Path]:
        """Find the checkpoint with the best performance metric."""
        checkpoints = self.get_checkpoint_files("checkpoint_step_*.pth")
        
        best_checkpoint = None
        best_score = -float('inf')
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                performance = checkpoint.get('performance_summary', {})
                score = performance.get(metric, -float('inf'))
                
                if score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint_path
                    
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
                continue
        
        return best_checkpoint


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

def get_latest_checkpoint(pattern: str = "checkpoint_step_*.pth") -> Optional[Path]:
    """Get the most recent checkpoint file."""
    return get_file_manager().get_latest_checkpoint(pattern)

def get_best_checkpoint(metric: str = "avg_score") -> Optional[Path]:
    """Get the checkpoint with the best performance."""
    return get_file_manager().find_best_checkpoint(metric)

def cleanup_old_checkpoints(keep_last: int = 5) -> int:
    """Remove old checkpoint files."""
    return get_file_manager().cleanup_old_checkpoints(keep_last)

def safe_save_json(data: Dict[Any, Any], filepath: Union[str, Path]) -> bool:
    """Safely save data to JSON file."""
    return get_file_manager().safe_save_json(data, filepath)

def safe_load_json(filepath: Union[str, Path]) -> Optional[Dict[Any, Any]]:
    """Safely load JSON data."""
    return get_file_manager().safe_load_json(filepath)

def validate_checkpoint(filepath: Union[str, Path]) -> Tuple[bool, str]:
    """Validate checkpoint file."""
    return get_file_manager().validate_checkpoint_file(filepath)


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    device_manager = get_device_manager()
    
    info = {
        'device_info': device_manager.device_info,
        'memory_usage': device_manager.get_memory_usage(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    return info


def setup_training_environment(config: Any = None) -> torch.device:
    """Setup comprehensive training environment."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ensure directories
    ensure_project_directories()
    
    # Set up device
    device = setup_device_for_training()
    
    # Set seeds for reproducibility
    seed = getattr(config, 'seed', 42) if config else 42
    set_seed(seed)
    
    # Log system info
    system_info = get_system_info()
    logger.info("🔧 Training environment setup complete")
    logger.info(f"Device: {device}")
    logger.info(f"Memory: {system_info['memory_percent']:.1f}% used")
    logger.info(f"CPU: {system_info['cpu_percent']:.1f}% used")
    
    return device


class ProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, total_episodes: int, log_every: int = 100):
        self.total_episodes = total_episodes
        self.log_every = log_every
        self.start_time = time.time()
        self.episode_times = deque(maxlen=100)
        
    def update(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Update progress tracking."""
        current_time = time.time()
        
        if episode > 0:
            episode_time = current_time - getattr(self, 'last_time', self.start_time)
            self.episode_times.append(episode_time)
        
        self.last_time = current_time
        
        if episode % self.log_every == 0:
            self._log_progress(episode, metrics, current_time)
    
    def _log_progress(self, episode: int, metrics: Dict[str, Any], current_time: float) -> None:
        """Log current progress."""
        elapsed = current_time - self.start_time
        progress = episode / self.total_episodes
        
        # Estimate remaining time
        if self.episode_times:
            avg_episode_time = sum(self.episode_times) / len(self.episode_times)
            remaining_episodes = self.total_episodes - episode
            eta = remaining_episodes * avg_episode_time
        else:
            eta = 0
        
        logger.info(
            f"Episode {episode:,}/{self.total_episodes:,} ({progress:.1%}) | "
            f"Score: {metrics.get('avg_score', 0):.2f} | "
            f"Elapsed: {format_time(elapsed)} | "
            f"ETA: {format_time(eta)}"
        )


# =============================================================================
# Model Utilities
# =============================================================================

def load_model_for_inference(checkpoint_path: Union[str, Path], device: torch.device = None) -> Tuple[torch.nn.Module, Dict]:
    """Load a model specifically for inference."""
    if device is None:
        device = get_device()
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Validate checkpoint
    is_valid, message = validate_checkpoint(checkpoint_path)
    if not is_valid:
        raise ValueError(f"Invalid checkpoint: {message}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state and metadata
    if 'online_net' in checkpoint:  # New format (AdvancedDQNAgent)
        from enhanced_dqn_agent import DuelingDoubleDQN
        config = checkpoint.get('config', {})
        model = DuelingDoubleDQN(
            input_dim=config.get('state_dim', 150),  # Default for 12x12 grid
            output_dim=config.get('action_dim', 4)
        ).to(device)
        model.load_state_dict(checkpoint['online_net'])
        
    elif 'model' in checkpoint:     # Old format (basic DQN)
        # Try to determine architecture from state dict
        state_dict = checkpoint['model']
        
        # Check if it's enhanced DQN (has fc4 layer)
        if 'fc4.weight' in state_dict:
            # Enhanced DQN from old training script
            class EnhancedDQN(torch.nn.Module):
                def __init__(self, input_dim: int, output_dim: int):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_dim, 512)
                    self.fc2 = torch.nn.Linear(512, 256) 
                    self.fc3 = torch.nn.Linear(256, 128)
                    self.fc4 = torch.nn.Linear(128, output_dim)
                    self.dropout = torch.nn.Dropout(0.1)
                
                def forward(self, x):
                    x = torch.nn.functional.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = torch.nn.functional.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = torch.nn.functional.relu(self.fc3(x))
                    x = self.fc4(x)
                    return x
            
            # Infer input dimension from first layer
            input_dim = state_dict['fc1.weight'].shape[1]
            model = EnhancedDQN(input_dim=input_dim, output_dim=4).to(device)
            
        else:
            # Basic DQN
            from agent.dqn import DQN
            fc1_weight_shape = state_dict['fc1.weight'].shape
            input_dim = fc1_weight_shape[1]
            hidden_dim = fc1_weight_shape[0]
            
            model = DQN(
                input_dim=input_dim,
                output_dim=4,
                hidden_dim=hidden_dim
            ).to(device)
        
        model.load_state_dict(state_dict)
    
    else:
        raise ValueError("Unknown checkpoint format")
    
    model.eval()
    
    # Extract metadata
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'device': str(device),
        'architecture': checkpoint.get('architecture', 'unknown'),
        'episode': checkpoint.get('episode', checkpoint.get('episodes', 'unknown')),
        'performance': checkpoint.get('performance_summary', {}),
        'config': checkpoint.get('config', {})
    }
    
    logger.info(f"✅ Model loaded for inference: {metadata['architecture']} from episode {metadata['episode']}")
    
    return model, metadata


import time
import sys