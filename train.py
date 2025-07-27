#!/usr/bin/env python3
"""
Training pipeline for SlytherNN - Professional DQN training with RTX 3090 optimizations.
Trains a new AI agent using the upgraded architecture and modern ML engineering practices.
"""

import os
import random
import numpy as np
import time
import torch
import torch.nn.functional as F
import csv
import datetime
import logging

# Professional architecture imports
from config import TrainingConfig, GameConfig
from agent import DQN, PrioritizedReplayMemory
from snake_game import VectorEnv
from utils import setup_device_for_training, get_optimal_batch_size, ensure_project_directories

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_enhanced_networks(state_dim: int, training_config: TrainingConfig, device: torch.device) -> tuple:
    """Create enhanced networks with deeper architecture for superior performance."""
    
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
            
            # Initialize weights
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.nn.functional.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    logger.info("🧠 Creating ENHANCED DQN with deeper architecture")
    
    policy_net = EnhancedDQN(input_dim=state_dim, output_dim=4).to(device)
    target_net = EnhancedDQN(input_dim=state_dim, output_dim=4).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    param_count = sum(p.numel() for p in policy_net.parameters())
    logger.info(f"📊 Enhanced network parameters: {param_count:,}")
    logger.info(f"🚀 Expected performance: >>25 score (human-level+)")
    
    return policy_net, target_net


def setup_training_environment(training_config: TrainingConfig) -> torch.device:
    """Set up training environment with proper seeding and device optimization."""
    # Seed everything for reproducibility
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    
    # Setup device with RTX 3090 optimizations
    device = setup_device_for_training()
    
    # Ensure directories exist
    ensure_project_directories()
    os.makedirs(training_config.log_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    return device


def select_actions_batch(model: torch.nn.Module, states: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Batch epsilon-greedy action selection optimized for RTX 3090."""
    batch_size = states.size(0)
    random_actions = torch.randint(0, 4, (batch_size,), device=states.device)
    
    with torch.no_grad():
        q_values = model(states)
        best_actions = torch.argmax(q_values, dim=1)
    
    probs = torch.rand(batch_size, device=states.device)
    chosen_actions = torch.where(probs < epsilon, random_actions, best_actions)
    return chosen_actions


def optimize_model(
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    memory: PrioritizedReplayMemory,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    scaler: torch.cuda.amp.GradScaler = None
) -> float:
    """Optimized model training step with mixed precision support."""
    # Get optimal batch size for RTX 3090
    batch_size = get_optimal_batch_size(training_config.batch_size)
    
    if len(memory) < batch_size:
        return None
    
    # Sample from prioritized replay
    batch, idxs, is_weights = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long, device=states.device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=states.device).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32, device=states.device).unsqueeze(1)
    is_weights = torch.tensor(is_weights, dtype=torch.float32, device=states.device).unsqueeze(1)

    # Mixed precision training for RTX 3090
    if scaler and training_config.use_mixed_precision:
        with torch.cuda.amp.autocast():
            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                expected_q = rewards + (training_config.gamma * next_q * (1 - dones))
            
            td_errors = current_q - expected_q
            loss = (is_weights * td_errors.pow(2)).mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        current_q = policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_net(next_states).max(1, keepdim=True)[0]
            expected_q = rewards + (training_config.gamma * next_q * (1 - dones))
        
        td_errors = current_q - expected_q
        loss = (is_weights * td_errors.pow(2)).mean()
        
        loss.backward()
        optimizer.step()
    
    optimizer.zero_grad()
    
    # Update priorities
    td_errors_np = td_errors.detach().abs().cpu().numpy().flatten()
    memory.update_priorities(idxs, td_errors_np)
    
    return loss.item()


def save_checkpoint(
    episode: int,
    policy_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    additional_data: dict = None
):
    """Save training checkpoint with metadata."""
    checkpoint_path = os.path.join(
        training_config.checkpoint_dir,
        f'dqn_snake_checkpoint_ep{episode}.pth'
    )
    
    checkpoint_data = {
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episode,
        'architecture': 'enhanced_dqn',  # Mark as enhanced architecture
        'timestamp': datetime.datetime.now().isoformat(),
        'training_config': training_config.__dict__
    }
    
    if additional_data:
        checkpoint_data.update(additional_data)
    
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Cleanup old checkpoints (keep last 3)
    cleanup_old_checkpoints(training_config.checkpoint_dir)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Remove old checkpoint files, keeping only the most recent."""
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('dqn_snake_checkpoint_ep') and f.endswith('.pth')]
    if len(files) <= keep_last:
        return
    
    # Sort by episode number
    files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
    files_to_remove = files[:-keep_last]
    
    for filename in files_to_remove:
        try:
            os.remove(os.path.join(checkpoint_dir, filename))
            logger.debug(f"🗑️ Removed old checkpoint: {filename}")
        except Exception as e:
            logger.warning(f"Failed to remove {filename}: {e}")


def main():
    """Main training loop with professional architecture integration."""
    logger.info("🚀 Starting SlytherNN training with ENHANCED architecture")
    
    # Load configurations
    training_config = TrainingConfig()
    game_config = GameConfig()
    
    # Setup training environment
    device = setup_training_environment(training_config)
    
    # Create vectorized environments
    logger.info(f"🎮 Creating {training_config.num_envs} parallel environments")
    envs = VectorEnv(num_envs=training_config.num_envs, device=device)
    state_dim = envs.get_states().shape[1]
    
    # ✅ FIXED: Use enhanced networks instead of basic networks
    policy_net, target_net = create_enhanced_networks(state_dim, training_config, device)
    
    # Setup optimizer and memory
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=training_config.learning_rate)
    memory = PrioritizedReplayMemory(capacity=training_config.memory_size)
    
    # Mixed precision scaler for RTX 3090
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and training_config.use_mixed_precision else None
    if scaler:
        logger.info("⚡ Mixed precision training enabled for RTX 3090")
    
    # Training variables
    epsilon = training_config.epsilon_start
    step_count = 0
    start_time = time.time()
    
    # Setup logging
    log_file = os.path.join(training_config.log_dir, f'training_log_enhanced_{int(time.time())}.csv')
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'EnvID', 'Reward', 'Steps', 'Epsilon', 'Loss', 'Timestamp'])
        
        # Initialize environments
        states = envs.reset()
        episode_rewards = torch.zeros(training_config.num_envs, device=device)
        episode_steps = torch.zeros(training_config.num_envs, device=device)
        episode_counts = torch.zeros(training_config.num_envs, device=device)
        
        logger.info("🏃 Starting enhanced training loop...")
        
        while episode_counts.min() < training_config.num_episodes:
            # Select actions
            actions = select_actions_batch(policy_net, states, epsilon)
            
            # Step environments
            next_states, rewards, dones = envs.step(actions)
            
            # Store experiences
            for i in range(training_config.num_envs):
                memory.add((
                    states[i].to(device),
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i].to(device),
                    dones[i].item()
                ))
            
            # Update tracking
            episode_rewards += rewards
            episode_steps += 1
            
            # Optimize model
            loss = optimize_model(policy_net, target_net, memory, optimizer, training_config, scaler)
            
            # Handle episode completions
            for i in range(training_config.num_envs):
                if dones[i]:
                    episode_counts[i] += 1
                    total_episodes = int(episode_counts.sum())
                    
                    # Log episode data
                    writer.writerow([
                        total_episodes,
                        i,
                        episode_rewards[i].item(),
                        int(episode_steps[i].item()),
                        epsilon,
                        loss if loss else 0.0,
                        datetime.datetime.now().isoformat()
                    ])
                    
                    # Progress logging with enhanced info
                    if total_episodes % 100 == 0:
                        avg_reward = episode_rewards[dones].mean().item() if dones.sum() > 0 else 0
                        logger.info(f"📊 Episode {total_episodes}: avg_reward={avg_reward:.2f}, epsilon={epsilon:.3f} [ENHANCED]")
                    
                    # Save checkpoint
                    if total_episodes % training_config.save_every == 0:
                        save_checkpoint(total_episodes, policy_net, optimizer, training_config)
                    
                    # Reset episode tracking
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
            
            # Update target network
            if step_count % training_config.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                logger.debug(f"🎯 Target network updated at step {step_count}")
            
            # Decay epsilon
            epsilon = max(training_config.epsilon_end, epsilon * training_config.epsilon_decay)
            
            # Update states and step count
            states = next_states
            step_count += training_config.num_envs
    
    # Training completed
    elapsed = time.time() - start_time
    total_episodes = int(episode_counts.sum())
    logger.info(f"🎉 Enhanced training completed!")
    logger.info(f"📊 Total episodes: {total_episodes}")
    logger.info(f"⏱️ Training time: {elapsed:.2f} seconds")
    logger.info(f"🚀 Average episodes/sec: {total_episodes/elapsed:.2f}")
    
    # Save final checkpoint
    save_checkpoint(total_episodes, policy_net, optimizer, training_config, {
        'training_completed': True,
        'total_training_time': elapsed,
        'architecture': 'enhanced_dqn'
    })


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

