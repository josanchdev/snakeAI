"""
AI Manager for model lifecycle and inference operations.
Handles model loading, prediction, and metadata management.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import logging

from config import ACTIONS, ACTION_NAMES
from utils import get_device, get_latest_checkpoint

logger = logging.getLogger(__name__)


class AIManager:
    """Manages AI model lifecycle and inference operations."""
    
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.device = get_device()
        self.model: Optional[torch.nn.Module] = None
        self.model_metadata: Dict[str, Any] = {}
        self.is_loaded = False
        
        # Calculate input dimension based on game state representation
        # grid_size * grid_size (board) + 4 (direction) + 2 (food relative position)
        self.input_dim = grid_size * grid_size + 4 + 2
        
    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load AI model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            # Get checkpoint path
            if checkpoint_path is None:
                checkpoint_file = get_latest_checkpoint()
                if checkpoint_file is None:
                    logger.error("No checkpoint files found")
                    return False
                checkpoint_path = str(checkpoint_file)
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            # Load checkpoint
            logger.info(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Import DQN here to avoid circular imports
            from agent.dqn import DQN
            
            # Initialize model
            self.model = DQN(input_dim=self.input_dim, output_dim=len(ACTIONS)).to(self.device)
            
            # Load model state
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                # Load additional metadata if available
                self.model_metadata = {
                    'checkpoint_path': str(checkpoint_path),
                    'episode': self._extract_episode_from_filename(checkpoint_path.name),
                    'optimizer_state': 'optimizer' in checkpoint,
                    'additional_data': list(checkpoint.keys())
                }
            else:
                # Legacy format - just model state dict
                self.model.load_state_dict(checkpoint)
                self.model_metadata = {
                    'checkpoint_path': str(checkpoint_path),
                    'episode': self._extract_episode_from_filename(checkpoint_path.name),
                    'format': 'legacy'
                }
            
            # Set to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully from episode {self.model_metadata.get('episode', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.is_loaded = False
            return False
    
    def predict_action(self, state: torch.Tensor, return_q_values: bool = False) -> Union[int, Tuple[int, torch.Tensor]]:
        """
        Predict the best action for a given state.
        
        Args:
            state: Game state tensor
            return_q_values: Whether to return Q-values along with action
            
        Returns:
            Action index (and optionally Q-values)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure state is on correct device and has batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            # Get Q-values
            with torch.no_grad():
                q_values = self.model(state)
                action_idx = torch.argmax(q_values, dim=1).item()
            
            if return_q_values:
                return action_idx, q_values.squeeze()
            return action_idx
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return random action as fallback
            import random
            return random.randint(0, len(ACTIONS) - 1)
    
    def get_action_probabilities(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities using softmax with temperature.
        
        Args:
            state: Game state tensor
            temperature: Temperature for softmax (higher = more random)
            
        Returns:
            Action probabilities tensor
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure state is on correct device and has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state)
            probabilities = F.softmax(q_values / temperature, dim=1)
            
        return probabilities.squeeze()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "device": str(self.device),
            "input_dim": self.input_dim,
            "output_dim": len(ACTIONS),
            "grid_size": self.grid_size,
            "actions": ACTION_NAMES,
            **self.model_metadata
        }
        
        # Add model parameter count
        if self.model is not None:
            info["parameters"] = sum(p.numel() for p in self.model.parameters())
            info["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info
    
    def analyze_state(self, state: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze a game state and provide detailed AI reasoning.
        
        Args:
            state: Game state tensor
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_loaded or self.model is None:
            return {"error": "Model not loaded"}
        
        action_idx, q_values = self.predict_action(state, return_q_values=True)
        probabilities = self.get_action_probabilities(state)
        
        # Convert to numpy for easier handling
        q_values_np = q_values.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        
        analysis = {
            "recommended_action": {
                "index": action_idx,
                "name": ACTION_NAMES[action_idx],
                "direction": ACTIONS[action_idx]
            },
            "q_values": {
                ACTION_NAMES[i]: float(q_values_np[i]) 
                for i in range(len(ACTION_NAMES))
            },
            "action_probabilities": {
                ACTION_NAMES[i]: float(probs_np[i]) 
                for i in range(len(ACTION_NAMES))
            },
            "confidence": float(torch.max(probabilities).item()),
            "uncertainty": float(1.0 - torch.max(probabilities).item())
        }
        
        return analysis
    
    def _extract_episode_from_filename(self, filename: str) -> Optional[int]:
        """Extract episode number from checkpoint filename."""
        try:
            if '_ep' in filename:
                return int(filename.split('_ep')[1].split('.')[0])
        except (IndexError, ValueError):
            pass
        return None
    
    def unload_model(self) -> None:
        """Unload the current model and free memory."""
        self.model = None
        self.model_metadata = {}
        self.is_loaded = False
        
        # Clear CUDA cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully")


# Global AI manager instance
_ai_manager = None


def get_ai_manager(grid_size: int = 12) -> AIManager:
    """Get the global AI manager instance."""
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = AIManager(grid_size)
    return _ai_manager

