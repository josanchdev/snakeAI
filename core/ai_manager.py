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
        Load AI model from checkpoint with automatic architecture detection.
        
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
            
            # Extract state dict and metadata
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                metadata = checkpoint
            else:
                # Legacy format - just model state dict
                state_dict = checkpoint
                metadata = {}
            
            # ✅ ENHANCED ARCHITECTURE DETECTION
            # Check if this is an enhanced model by looking for fc4 layer
            is_enhanced = 'fc4.weight' in state_dict
            
            if is_enhanced:
                logger.info("🧠 Detected Enhanced DQN architecture")
                
                # Create Enhanced DQN class (same as in train.py)
                class EnhancedDQN(torch.nn.Module):
                    """Enhanced DQN with deeper architecture and better regularization."""
                    
                    def __init__(self, input_dim: int, output_dim: int):
                        super().__init__()
                        # Deeper network for better pattern recognition
                        self.fc1 = torch.nn.Linear(input_dim, 512)
                        self.fc2 = torch.nn.Linear(512, 256) 
                        self.fc3 = torch.nn.Linear(256, 128)
                        self.fc4 = torch.nn.Linear(128, output_dim)
                        
                        # Dropout for better generalization
                        self.dropout = torch.nn.Dropout(0.1)
                    
                    def forward(self, x):
                        x = torch.nn.functional.relu(self.fc1(x))
                        x = self.dropout(x)
                        x = torch.nn.functional.relu(self.fc2(x))
                        x = self.dropout(x)
                        x = torch.nn.functional.relu(self.fc3(x))
                        x = self.fc4(x)
                        return x
                
                # Initialize enhanced model
                self.model = EnhancedDQN(input_dim=self.input_dim, output_dim=len(ACTIONS)).to(self.device)
                
            else:
                logger.info("🔧 Detected Basic DQN architecture")
                # Import basic DQN
                from agent.dqn import DQN
                
                # Auto-detect hidden dimension from checkpoint
                fc1_weight_shape = state_dict['fc1.weight'].shape
                hidden_dim = fc1_weight_shape[0]  # First dimension is output size (hidden_dim)
                
                logger.info(f"🔍 Detected hidden_dim={hidden_dim}")
                
                # Initialize model with detected architecture
                self.model = DQN(
                    input_dim=self.input_dim, 
                    output_dim=len(ACTIONS),
                    hidden_dim=hidden_dim
                ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(state_dict)
            
            # Set comprehensive metadata
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_metadata = {
                'checkpoint_path': str(checkpoint_path),
                'episode': self._extract_episode_from_filename(checkpoint_path.name),
                'architecture': 'enhanced' if is_enhanced else 'basic',
                'parameters': param_count,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device),
                'input_dim': self.input_dim,
                'output_dim': len(ACTIONS),
                'format': 'auto_detected',
                'timestamp': metadata.get('timestamp', 'unknown'),
                'training_completed': metadata.get('training_completed', False)
            }
            
            # Set to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"✅ Model loaded successfully from episode {self.model_metadata.get('episode', 'unknown')}")
            logger.info(f"🏗️ Architecture: {self.model_metadata['architecture']} ({param_count:,} parameters)")
            
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
        """Get comprehensive information about the loaded model."""
        if not self.is_loaded:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "grid_size": self.grid_size,
            "actions": ACTION_NAMES,
            **self.model_metadata
        }
        
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
            "uncertainty": float(1.0 - torch.max(probabilities).item()),
            "model_info": {
                "architecture": self.model_metadata.get('architecture', 'unknown'),
                "parameters": self.model_metadata.get('parameters', 0),
                "episode": self.model_metadata.get('episode', 'unknown')
            }
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

