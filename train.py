#!/usr/bin/env python3
"""
Enhanced Training Pipeline for SlytherNN
Features: Smart reward shaping, curriculum learning, checkpoint resuming, comprehensive logging
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from collections import deque

# Project imports (adjust based on final structure)
from enhanced_dqn_agent import AdvancedDQNAgent, ACTIONS
from snake_game.vector_env import VectorEnv
from utils import setup_device_for_training, ensure_project_directories

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Enhanced training configuration with smart defaults."""
    
    # Environment settings
    num_envs: int = 96
    grid_size: int = 12
    max_episodes: int = 5000
    max_steps_per_episode: int = 500
    
    # Network settings
    learning_rate: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 200000
    hidden_dim: int = 512
    
    # Training parameters
    gamma: float = 0.99
    target_update_freq: int = 2000
    n_step: int = 3
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 4
    episodes_per_stage: int = 1000
    
    # Logging and checkpointing
    log_every: int = 50
    save_every: int = 250
    eval_every: int = 100
    eval_episodes: int = 10
    
    # Device settings
    device: str = "auto"
    mixed_precision: bool = True
    
    # Checkpoint management
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_from: Optional[str] = None


class SmartRewardShaper:
    """
    Intelligent reward shaping system to overcome learning plateaus.
    Uses potential-based shaping and curriculum learning.
    """
    
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.prev_potential = {}
        
        # Reward coefficients (tunable)
        self.rewards = {
            'food': 10.0,           # Eating food
            'death': -10.0,         # Dying
            'step': -0.01,          # Living penalty (encourages efficiency)
            'approach_food': 1.0,   # Getting closer to food
            'survive': 0.1,         # Staying alive
            'wall_danger': -0.5,    # Getting close to walls
            'self_danger': -1.0,    # Risk of self-collision
            'efficiency': 1.0,      # Moving toward food efficiently
        }
    
    def compute_shaped_reward(
        self, 
        env_id: int,
        prev_state: torch.Tensor, 
        action: int, 
        reward: float,
        next_state: torch.Tensor, 
        done: bool,
        game_info: Dict
    ) -> float:
        """Compute shaped reward using multiple heuristics."""
        
        shaped_reward = reward  # Base reward
        
        if done:
            if reward > 0:  # Won/ate food
                return shaped_reward
            else:  # Died
                return self.rewards['death']
        
        # Extract game information
        snake_head = game_info.get('snake_head', (0, 0))
        food_pos = game_info.get('food_pos', (0, 0))
        snake_length = game_info.get('snake_length', 3)
        
        # 1. Distance-based reward (potential shaping)
        current_distance = abs(snake_head[0] - food_pos[0]) + abs(snake_head[1] - food_pos[1])
        current_potential = -current_distance / self.grid_size
        
        if env_id in self.prev_potential:
            distance_reward = self.rewards['approach_food'] * (current_potential - self.prev_potential[env_id])
            shaped_reward += distance_reward
        
        self.prev_potential[env_id] = current_potential
        
        # 2. Survival bonus (encourages longer games)
        shaped_reward += self.rewards['survive']
        
        # 3. Danger penalties
        wall_penalty = self._compute_wall_danger(snake_head)
        self_collision_penalty = self._compute_self_danger(snake_head, game_info.get('snake_body', []))
        
        shaped_reward += wall_penalty + self_collision_penalty
        
        # 4. Efficiency bonus (straight line movement toward food)
        efficiency_bonus = self._compute_efficiency_bonus(snake_head, food_pos, action)
        shaped_reward += efficiency_bonus
        
        return shaped_reward
    
    def _compute_wall_danger(self, head: Tuple[int, int]) -> float:
        """Penalize getting close to walls."""
        x, y = head
        min_dist_to_wall = min(x, y, self.grid_size - 1 - x, self.grid_size - 1 - y)
        
        if min_dist_to_wall <= 1:
            return self.rewards['wall_danger']
        return 0.0
    
    def _compute_self_danger(self, head: Tuple[int, int], body: List[Tuple[int, int]]) -> float:
        """Penalize getting close to own body."""
        if not body:
            return 0.0
        
        for segment in body[1:4]:  # Check next 3 segments
            distance = abs(head[0] - segment[0]) + abs(head[1] - segment[1])
            if distance <= 1:
                return self.rewards['self_danger']
        return 0.0
    
    def _compute_efficiency_bonus(self, head: Tuple[int, int], food: Tuple[int, int], action: int) -> float:
        """Reward moving directly toward food."""
        dx = food[0] - head[0]
        dy = food[1] - head[1]
        
        # Desired action based on food position
        if abs(dx) > abs(dy):
            desired_action = 3 if dx > 0 else 2  # RIGHT or LEFT
        else:
            desired_action = 1 if dy > 0 else 0  # DOWN or UP
        
        if action == desired_action:
            return self.rewards['efficiency'] * 0.1  # Small bonus for good moves
        return 0.0


class CurriculumManager:
    """Manages curriculum learning stages for progressive difficulty."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.episodes_in_stage = 0
        
        # Define curriculum stages
        self.stages = [
            {'grid_size': 8, 'max_steps': 200, 'description': 'Small grid, basic learning'},
            {'grid_size': 10, 'max_steps': 300, 'description': 'Medium grid, longer episodes'},
            {'grid_size': 12, 'max_steps': 400, 'description': 'Standard grid, full difficulty'},
            {'grid_size': 12, 'max_steps': 500, 'description': 'Extended play, mastery stage'},
        ]
        
        logger.info(f"Curriculum learning initialized with {len(self.stages)} stages")
    
    def get_current_settings(self) -> Dict:
        """Get current curriculum stage settings."""
        if not self.config.use_curriculum:
            return {'grid_size': self.config.grid_size, 'max_steps': self.config.max_steps_per_episode}
        
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def should_advance_stage(self, performance_metrics: Dict) -> bool:
        """Determine if we should advance to next curriculum stage."""
        if not self.config.use_curriculum:
            return False
        
        self.episodes_in_stage += 1
        
        # Advance based on episodes or performance
        episodes_threshold = self.config.episodes_per_stage
        performance_threshold = performance_metrics.get('avg_score', 0) > 5  # Adjust threshold
        
        if self.episodes_in_stage >= episodes_threshold or performance_threshold:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.episodes_in_stage = 0
                logger.info(f"🎓 Advanced to curriculum stage {self.current_stage}: {self.stages[self.current_stage]['description']}")
                return True
        
        return False


class TrainingLogger:
    """Comprehensive training logger with metrics tracking and visualization."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.episode_metrics = []
        self.training_metrics = []
        
        # Running averages
        self.score_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=100)
        
        # Best performance tracking
        self.best_avg_score = -float('inf')
        self.best_model_path = None
        
        # Setup log files
        self.setup_log_files()
    
    def setup_log_files(self):
        """Initialize log files with headers."""
        # Episode log
        self.episode_log_path = self.log_dir / 'episodes.csv'
        with open(self.episode_log_path, 'w') as f:
            f.write('episode,env_id,score,steps,reward,stage,timestamp\n')
        
        # Training log
        self.training_log_path = self.log_dir / 'training.csv'
        with open(self.training_log_path, 'w') as f:
            f.write('step,loss,avg_score,best_score,learning_rate,stage,timestamp\n')
        
        # Config log
        config_path = self.log_dir / 'config.json'
        # Will be filled when training starts
    
    def log_episode(self, episode: int, env_id: int, score: int, steps: int, 
                   total_reward: float, stage: int):
        """Log individual episode data."""
        timestamp = time.time()
        
        # Update running metrics
        self.score_window.append(score)
        
        # Log to file
        with open(self.episode_log_path, 'a') as f:
            f.write(f'{episode},{env_id},{score},{steps},{total_reward:.3f},{stage},{timestamp}\n')
        
        # Store in memory
        self.episode_metrics.append({
            'episode': episode,
            'env_id': env_id,
            'score': score,
            'steps': steps,
            'reward': total_reward,
            'stage': stage,
            'timestamp': timestamp
        })
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, stage: int):
        """Log training step data."""
        if loss is not None:
            self.loss_window.append(loss)
        
        avg_score = np.mean(self.score_window) if self.score_window else 0
        best_score = max(self.score_window) if self.score_window else 0
        
        # Check for new best performance
        if avg_score > self.best_avg_score:
            self.best_avg_score = avg_score
        
        timestamp = time.time()
        
        # Log to file
        with open(self.training_log_path, 'a') as f:
            f.write(f'{step},{loss or 0:.6f},{avg_score:.3f},{best_score},{learning_rate:.6f},{stage},{timestamp}\n')
        
        # Store in memory
        self.training_metrics.append({
            'step': step,
            'loss': loss,
            'avg_score': avg_score,
            'best_score': best_score,
            'learning_rate': learning_rate,
            'stage': stage,
            'timestamp': timestamp
        })
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.score_window:
            return {'avg_score': 0, 'best_score': 0, 'episodes': 0}
        
        return {
            'avg_score': np.mean(self.score_window),
            'best_score': max(self.score_window),
            'std_score': np.std(self.score_window),
            'episodes': len(self.episode_metrics),
            'best_avg_score': self.best_avg_score
        }
    
    def save_config(self, config: TrainingConfig):
        """Save training configuration."""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)


class EnhancedTrainer:
    """Main training class with all enhancements."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize components
        self.reward_shaper = SmartRewardShaper(config.grid_size)
        self.curriculum = CurriculumManager(config)
        self.logger = TrainingLogger(config.log_dir)
        
        # Training state
        self.global_step = 0
        self.total_episodes = 0
        self.start_time = time.time()
        
        # Setup directories
        ensure_project_directories()
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        logger.info("🚀 Enhanced Trainer initialized")
    
    def _setup_device(self) -> torch.device:
        """Setup training device with optimizations."""
        if self.config.device == "auto":
            device = setup_device_for_training()
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Training device: {device}")
        return device
    
    def _create_agent(self, state_dim: int) -> AdvancedDQNAgent:
        """Create the DQN agent."""
        return AdvancedDQNAgent(
            state_dim=state_dim,
            action_dim=len(ACTIONS),
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            target_update_freq=self.config.target_update_freq,
            n_step=self.config.n_step,
            device=self.device
        )
    
    def _create_environments(self, grid_size: int, max_steps: int) -> VectorEnv:
        """Create vectorized environments."""
        return VectorEnv(
            num_envs=self.config.num_envs,
            grid_size=grid_size,
            device=self.device
        )
    
    def _extract_game_info(self, env, env_id: int) -> Dict:
        """Extract game information for reward shaping."""
        game = env.envs[env_id]
        return {
            'snake_head': game.snake.head(),
            'food_pos': game.fruit.position,
            'snake_length': len(game.snake.body),
            'snake_body': game.snake.body
        }
    
    def train(self):
        """Main training loop."""
        logger.info("🎯 Starting enhanced training...")
        
        # Save config
        self.logger.save_config(self.config)
        
        # Initialize with curriculum settings
        curr_settings = self.curriculum.get_current_settings()
        envs = self._create_environments(curr_settings['grid_size'], curr_settings['max_steps'])
        
        # Create agent
        state_dim = envs.get_states().shape[1]
        agent = self._create_agent(state_dim)
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._resume_training(agent)
        
        # Training loop
        states = envs.reset()
        episode_rewards = torch.zeros(self.config.num_envs, device=self.device)
        episode_steps = torch.zeros(self.config.num_envs, device=self.device)
        
        logger.info(f"🏃 Training started with {self.config.num_envs} environments")
        
        while self.total_episodes < self.config.max_episodes:
            # Select actions
            actions = []
            for i in range(self.config.num_envs):
                action = agent.select_action(states[i])
                actions.append(action)
            actions = torch.tensor(actions, device=self.device)
            
            # Store previous states for reward shaping
            prev_states = states.clone()
            
            # Step environments
            next_states, base_rewards, dones = envs.step(actions)
            
            # Apply reward shaping
            shaped_rewards = []
            for i in range(self.config.num_envs):
                game_info = self._extract_game_info(envs, i)
                shaped_reward = self.reward_shaper.compute_shaped_reward(
                    i, prev_states[i], actions[i].item(), base_rewards[i].item(),
                    next_states[i], dones[i].item(), game_info
                )
                shaped_rewards.append(shaped_reward)
            
            shaped_rewards = torch.tensor(shaped_rewards, device=self.device)
            episode_rewards += shaped_rewards
            episode_steps += 1
            
            # Store transitions
            for i in range(self.config.num_envs):
                agent.store_transition(
                    prev_states[i], actions[i].item(), shaped_rewards[i].item(),
                    next_states[i], dones[i].item()
                )
            
            # Update agent
            loss = agent.update()
            
            # Handle episode completions
            for i in range(self.config.num_envs):
                if dones[i]:
                    # Log episode
                    score = envs.envs[i].score
                    self.logger.log_episode(
                        self.total_episodes, i, score, int(episode_steps[i].item()),
                        float(episode_rewards[i].item()), self.curriculum.current_stage
                    )
                    
                    # Update counters
                    self.total_episodes += 1
                    agent.episodes += 1
                    
                    # Reset episode tracking
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
            
            # Logging and checkpointing
            if self.global_step % self.config.log_every == 0:
                self._log_training_progress(agent, loss)
            
            if self.global_step % self.config.save_every == 0:
                self._save_checkpoint(agent)
            
            if self.global_step % self.config.eval_every == 0:
                self._evaluate_agent(agent, envs)
            
            # Check curriculum advancement
            if self.global_step % 1000 == 0:  # Check every 1000 steps
                performance = self.logger.get_performance_summary()
                if self.curriculum.should_advance_stage(performance):
                    # Recreate environments with new settings
                    curr_settings = self.curriculum.get_current_settings()
                    envs = self._create_environments(curr_settings['grid_size'], curr_settings['max_steps'])
                    states = envs.reset()
                    episode_rewards = torch.zeros(self.config.num_envs, device=self.device)
                    episode_steps = torch.zeros(self.config.num_envs, device=self.device)
            
            # Update states and step counter
            states = next_states
            self.global_step += 1
        
        # Training completed
        self._training_completed(agent)
    
    def _log_training_progress(self, agent: AdvancedDQNAgent, loss: Optional[float]):
        """Log training progress."""
        performance = self.logger.get_performance_summary()
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        self.logger.log_training_step(
            self.global_step, loss, current_lr, self.curriculum.current_stage
        )
        
        if self.global_step % (self.config.log_every * 10) == 0:  # Less frequent console output
            elapsed = time.time() - self.start_time
            logger.info(
                f"Step {self.global_step:,} | Episodes: {self.total_episodes:,} | "
                f"Avg Score: {performance['avg_score']:.2f} | Best: {performance['best_score']} | "
                f"Loss: {loss:.6f if loss else 0:.6f} | Stage: {self.curriculum.current_stage} | "
                f"Time: {elapsed:.0f}s"
            )
    
    def _save_checkpoint(self, agent: AdvancedDQNAgent):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pth"
        
        additional_info = {
            'global_step': self.global_step,
            'total_episodes': self.total_episodes,
            'curriculum_stage': self.curriculum.current_stage,
            'curriculum_episodes_in_stage': self.curriculum.episodes_in_stage,
            'performance_summary': self.logger.get_performance_summary(),
            'training_time': time.time() - self.start_time
        }
        
        agent.save(str(checkpoint_path), additional_info)
        
        # Clean up old checkpoints (keep last 3)
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pth"))
        
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        for checkpoint in checkpoints[:-keep_last]:
            try:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {checkpoint}: {e}")
    
    def _evaluate_agent(self, agent: AdvancedDQNAgent, envs: VectorEnv):
        """Evaluate agent performance."""
        # Set to deterministic mode
        agent.online_net.eval()
        
        eval_scores = []
        eval_steps = []
        
        for _ in range(self.config.eval_episodes):
            states = envs.reset()
            total_steps = 0
            
            while total_steps < 1000:  # Max eval steps
                actions = []
                for i in range(self.config.num_envs):
                    action = agent.select_action(states[i], deterministic=True)
                    actions.append(action)
                
                actions = torch.tensor(actions, device=self.device)
                states, rewards, dones = envs.step(actions)
                total_steps += 1
                
                if dones.any():
                    for i in range(self.config.num_envs):
                        if dones[i]:
                            eval_scores.append(envs.envs[i].score)
                            eval_steps.append(total_steps)
                    break
        
        # Back to training mode
        agent.online_net.train()
        
        if eval_scores:
            avg_eval_score = np.mean(eval_scores)
            agent.scheduler.step(avg_eval_score)  # Update learning rate based on performance
            
            logger.info(f"🎯 Evaluation - Avg Score: {avg_eval_score:.2f}, Scores: {eval_scores}")
    
    def _resume_training(self, agent: AdvancedDQNAgent):
        """Resume training from checkpoint."""
        logger.info(f"📁 Resuming training from {self.config.resume_from}")
        
        checkpoint_data = agent.load(self.config.resume_from)
        
        # Restore training state
        self.global_step = checkpoint_data.get('global_step', 0)
        self.total_episodes = checkpoint_data.get('total_episodes', 0)
        self.curriculum.current_stage = checkpoint_data.get('curriculum_stage', 0)
        self.curriculum.episodes_in_stage = checkpoint_data.get('curriculum_episodes_in_stage', 0)
        
        logger.info(f"✅ Resumed from step {self.global_step}, episode {self.total_episodes}")
    
    def _training_completed(self, agent: AdvancedDQNAgent):
        """Handle training completion."""
        elapsed = time.time() - self.start_time
        performance = self.logger.get_performance_summary()
        
        logger.info("🎉 Training completed!")
        logger.info(f"📊 Final Performance:")
        logger.info(f"   Episodes: {self.total_episodes:,}")
        logger.info(f"   Average Score: {performance['avg_score']:.2f}")
        logger.info(f"   Best Score: {performance['best_score']}")
        logger.info(f"   Training Time: {elapsed:.2f}s")
        
        # Save final model
        final_path = Path(self.config.checkpoint_dir) / "final_model.pth"
        agent.save(str(final_path), {
            'training_completed': True,
            'final_performance': performance,
            'total_training_time': elapsed
        })
        
        logger.info(f"💾 Final model saved to {final_path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Enhanced SlytherNN Training")
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--envs', type=int, default=96, help='Number of environments')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Override with command line args
    if args.episodes:
        config.max_episodes = args.episodes
    if args.envs:
        config.num_envs = args.envs
    if args.resume:
        config.resume_from = args.resume
    if args.no_curriculum:
        config.use_curriculum = False
    
    # Start training
    trainer = EnhancedTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()