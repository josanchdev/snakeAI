"""
Device utilities optimized for high-end hardware configurations.
Enhanced for RTX 3090 + Ryzen 9 5900X systems.
"""

import torch
import psutil
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device detection and optimization for training and inference."""
    
    def __init__(self):
        self.device = self._detect_optimal_device()
        self.device_info = self._get_device_info()
        self._apply_hardware_specific_optimizations()
        
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
        """Get detailed information about the current device."""
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
        """Get CUDA device information with high-end GPU detection."""
        props = torch.cuda.get_device_properties(self.device)
        memory_gb = props.total_memory / (1024**3)
        
        # Detect high-end configurations
        is_high_end = (
            memory_gb >= 20 or  # 20GB+ VRAM (RTX 3090/4090, etc.)
            props.major >= 8    # Ampere architecture or newer
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
            9: "Hopper"
        }
        return arch_map.get(major, f"Compute {major}.x")
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information with high-core-count detection."""
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
        """Get MPS device information."""
        return {
            "device_name": "Apple Silicon GPU",
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
    
    def _apply_hardware_specific_optimizations(self) -> None:
        """Apply optimizations based on detected hardware."""
        if self.device.type == "cuda" and self.device_info.get("is_high_end"):
            # High-end GPU optimizations
            torch.backends.cuda.matmul.allow_tf32 = True  # RTX 30/40 series optimization
            torch.backends.cudnn.allow_tf32 = True
            logger.info("🔥 Applied high-end GPU optimizations (TF32 enabled)")
    
    def optimize_for_training(self) -> None:
        """Apply device-specific optimizations for training."""
        if self.device.type == "cuda":
            # Enhanced CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Better performance
            
            # High-end GPU specific optimizations
            if self.device_info.get("is_high_end"):
                # Enable memory allocation optimizations for large VRAM
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of 24GB
                logger.info("🚀 Optimized for high-end GPU: Using 95% of VRAM")
            
            torch.cuda.empty_cache()
            logger.info("Applied CUDA optimizations for training")
            
        elif self.device.type == "cpu":
            # CPU optimizations for high-core-count systems
            cpu_cores = self.device_info.get("cpu_count", 4)
            if cpu_cores >= 8:
                # For 8+ core systems like Ryzen 9 5900X
                optimal_threads = min(cpu_cores, 16)  # Cap at 16 for diminishing returns
                torch.set_num_threads(optimal_threads)
                logger.info(f"🔥 High-end CPU detected: Using {optimal_threads} threads")
            else:
                torch.set_num_threads(min(4, cpu_cores))
                
            # Enable optimized CPU operations
            torch.set_num_interop_threads(2)
            logger.info("Applied CPU optimizations for training")
    
    def get_optimal_batch_size_recommendation(self, base_batch_size: int = 128) -> int:
        """Recommend optimal batch size based on available hardware."""
        if self.device.type == "cuda":
            memory_gb = self.device_info.get("memory_total_gb", 8)
            
            if memory_gb >= 20:  # RTX 3090/4090 territory
                return base_batch_size * 4  # 512 batch size
            elif memory_gb >= 16:
                return base_batch_size * 3  # 384 batch size
            elif memory_gb >= 12:
                return base_batch_size * 2  # 256 batch size
            else:
                return base_batch_size  # 128 batch size
        
        return base_batch_size
    
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
                "available_gb": total_gb - reserved_gb
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
        """Print detailed device information with performance tips."""
        print(f"\n🖥️  Hardware Configuration:")
        print(f"   Device: {self.device}")
        print(f"   Name: {self.device_info['device_name']}")
        
        if self.device.type == "cuda":
            memory_gb = self.device_info.get('memory_total_gb', 0)
            print(f"   VRAM: {memory_gb:.1f} GB")
            print(f"   Architecture: {self.device_info.get('architecture', 'Unknown')}")
            print(f"   Compute Capability: {self.device_info.get('compute_capability', 'Unknown')}")
            
            if self.device_info.get('is_high_end'):
                print(f"   🚀 High-End GPU Detected!")
                recommended_batch = self.get_optimal_batch_size_recommendation()
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


# Global optimized device manager
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
    """Setup and optimize device for training with hardware-specific optimizations."""
    manager = get_device_manager()
    manager.optimize_for_training()
    manager.print_device_info()
    return manager.device

def get_optimal_batch_size(base_batch_size: int = 128) -> int:
    """Get hardware-optimized batch size recommendation."""
    return get_device_manager().get_optimal_batch_size_recommendation(base_batch_size)

def get_memory_info() -> Dict[str, float]:
    """Get current device memory usage."""
    return get_device_manager().get_memory_usage()

def clear_device_memory() -> None:
    """Clear device memory cache."""
    get_device_manager().clear_memory()

