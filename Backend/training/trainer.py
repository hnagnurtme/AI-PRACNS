"""
Optimized Trainer for DuelingDQN Routing Agent
Training loop hiệu quả với advanced optimizations và monitoring
"""
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import json
import time
from collections import deque
import psutil

from agent.dueling_dqn import DuelingDQNAgent
from environment.routing_env import RoutingEnvironment
from environment.state_builder import RoutingStateBuilder
from config import Config
from training.training_dashboard import TrainingDashboard
from training.imitation_learning import ImitationLearning
from training.curriculum_learning import CurriculumScheduler

logger = logging.getLogger(__name__)


class RoutingTrainer:
    """
    Optimized trainer với advanced features và monitoring
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or Config.get_yaml_config()
        
        training_config = self.config.get('training', {})
        self.max_episodes = training_config.get('max_episodes', 5000)
        self.max_steps_per_episode = training_config.get('max_steps_per_episode', 15)
        self.eval_frequency = training_config.get('eval_frequency', 25)
        self.eval_episodes = training_config.get('eval_episodes', 20)
        self.save_frequency = training_config.get('save_frequency', 100)
        
        self.target_update_frequency = training_config.get('target_update_frequency', 100)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        self.early_stopping_patience = training_config.get('early_stopping_patience', 100)
        
        # Model paths
        rl_config = self.config.get('rl_agent', {})
        model_path_str = rl_config.get('model_path', './models/rl_agent')
        self.model_path = Path(model_path_str)
        
        if self.model_path.suffix:
            self.model_path = self.model_path.parent
            
        self.checkpoint_path = Path(rl_config.get('checkpoint_path', './models/checkpoints'))
        self.best_model_path = Path(rl_config.get('best_model_path', './models/best_models'))
        
        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        
        # Training state với optimizations
        self.episode = 0
        self.total_steps = 0
        self.best_mean_reward = float('-inf')
        self.early_stopping_counter = 0
        self.training_start_time = None
        
        # Metrics với rolling averages
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = deque(maxlen=500)
        self.epsilon_history = deque(maxlen=100)
        self.episode_success = deque(maxlen=100)
        self.episode_coverage = deque(maxlen=100)
        self.reward_per_step = deque(maxlen=100)
        
        # Performance monitoring
        self.episode_times = deque(maxlen=50)
        
        # Tensorboard
        self.tensorboard_enabled = training_config.get('tensorboard_enabled', True)
        tensorboard_log_dir_str = rl_config.get('tensorboard_log_dir', './logs/tensorboard')
        
        # Convert to absolute path to avoid issues with relative paths
        if not os.path.isabs(tensorboard_log_dir_str):
            # Get Backend root directory (parent of training directory)
            backend_root = Path(__file__).parent.parent
            self.tensorboard_log_dir = backend_root / tensorboard_log_dir_str.lstrip('./')
        else:
            self.tensorboard_log_dir = Path(tensorboard_log_dir_str)
        
        # Create directory if it doesn't exist
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = None
        
        if self.tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
                logger.info(f"Tensorboard logging to {self.tensorboard_log_dir.absolute()}")
            except ImportError:
                logger.warning("Tensorboard not available")
                self.tensorboard_enabled = False
        
        # Custom Training Dashboard
        dashboard_config = training_config.get('dashboard', {})
        self.dashboard_enabled = dashboard_config.get('enabled', True)
        self.dashboard = TrainingDashboard(
            config=self.config,
            log_dir=self.tensorboard_log_dir,
            auto_cleanup=dashboard_config.get('auto_cleanup_logs', True),
            retention_days=dashboard_config.get('log_retention_days', 7),
            update_frequency=dashboard_config.get('live_update_frequency', 10)
        )
        
        # Imitation Learning - config at root level
        il_config = self.config.get('imitation_learning', {})
        self.use_imitation_learning = il_config.get('enabled', True)
        self.il_warmup_episodes = il_config.get('warmup_episodes', 100)
        self.il_num_demos = il_config.get('num_demos', 500)
        self.imitation_learning = ImitationLearning(self.config) if self.use_imitation_learning else None
        
        # Curriculum Learning - config at root level (key is 'curriculum')
        cl_config = self.config.get('curriculum', {})
        self.use_curriculum_learning = cl_config.get('enabled', True)
        self.curriculum_scheduler = CurriculumScheduler(self.config) if self.use_curriculum_learning else None
        
        if self.use_imitation_learning:
            logger.info("🎓 Imitation Learning ENABLED")
        if self.use_curriculum_learning:
            logger.info("📚 Curriculum Learning ENABLED")
    
    def train(
        self,
        nodes: List[Dict],
        terminals: List[Dict],
        episodes: Optional[int] = None
    ):
        """Optimized training loop với advanced monitoring"""
        max_episodes = episodes or self.max_episodes
        self.training_start_time = time.time()
        
        # Initialize environment với optimized parameters
        env = RoutingEnvironment(
            nodes=nodes,
            terminals=terminals,
            config=self.config,
            max_steps=self.max_steps_per_episode
        )
        
        # Initialize state builder
        state_builder = RoutingStateBuilder(self.config)
        state_dim = state_builder.state_dimension
        action_dim = env.action_space.n
        
        # Initialize agent với optimized config
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.config
        )
        
        logger.info(f"Starting optimized training: {max_episodes} episodes")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Device: {agent.device}")
        logger.info(f"Memory usage: {self._get_memory_usage()} MB")
        
        # 🎓 Imitation Learning: Generate expert demonstrations
        if self.use_imitation_learning and self.imitation_learning:
            logger.info(f"🎓 Generating {self.il_num_demos} expert demonstrations...")
            try:
                self.imitation_learning.generate_comprehensive_demos(
                    terminals=terminals,
                    nodes=nodes,
                    num_demos=self.il_num_demos
                )
                logger.info(f"🎓 Generated {len(self.imitation_learning.expert_demos)} demos")
            except Exception as e:
                logger.warning(f"Failed to generate expert demos: {e}")
        
        # 📚 Curriculum Learning: Log initial level
        if self.use_curriculum_learning and self.curriculum_scheduler:
            level = self.curriculum_scheduler.get_current_level()
            logger.info(f"📚 Starting at curriculum level: {level.name} (difficulty: {level.difficulty})")
        
        # Start training dashboard
        if self.dashboard_enabled:
            self.dashboard.start(total_episodes=max_episodes)
        
        # Training loop với optimizations
        for episode in range(max_episodes):
            self.episode = episode
            episode_start_time = time.time()
            
            # Reset environment
            state, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            episode_losses = []
            
            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(state, deterministic=False)
                
                # Step environment
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                agent.replay_buffer.push(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                # Train agent với gradient clipping
                train_metrics = agent.train_step()
                if train_metrics:
                    episode_losses.append(train_metrics['loss'])
                    self.training_losses.append(train_metrics['loss'])
                elif agent.total_steps % 100 == 0 and len(agent.replay_buffer) < agent.learning_starts:
                    # Log buffer status periodically
                    logger.debug(
                        f"Buffer: {len(agent.replay_buffer)}/{agent.learning_starts} "
                        f"(training starts at {agent.learning_starts} experiences)"
                    )
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                agent.total_steps += 1
                self.total_steps += 1
                
                # Update epsilon và target network
                if agent.total_steps % self.target_update_frequency == 0:
                    agent.update_target_network()
                
                max_training_steps = max_episodes * self.max_steps_per_episode
                agent.update_epsilon(agent.total_steps, max_training_steps)
            
            # Episode complete - update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(agent.epsilon)
            
            # Calculate coverage (visited nodes / total nodes)
            visited_nodes = len(env.visited_nodes) if hasattr(env, 'visited_nodes') else 0
            total_nodes = len(nodes)
            coverage = visited_nodes / total_nodes if total_nodes > 0 else 0.0
            self.episode_coverage.append(coverage)
            
            # Calculate success
            success = 1 if terminated else 0
            self.episode_success.append(success)
            
            # Calculate reward per step
            reward_per_step = episode_reward / episode_length if episode_length > 0 else 0.0
            self.reward_per_step.append(reward_per_step)
            
            episode_time = time.time() - episode_start_time
            self.episode_times.append(episode_time)
            
            # Update custom dashboard
            if self.dashboard_enabled:
                mean_loss = np.mean(list(self.training_losses)[-10:]) if self.training_losses else 0.0
                self.dashboard.update(
                    episode=episode + 1,
                    reward=episode_reward,
                    success=bool(terminated),
                    loss=mean_loss,
                    epsilon=agent.epsilon,
                    extra_metrics={
                        'coverage': f"{coverage:.1%}",
                        'episode_length': episode_length,
                        'best_reward': self.best_mean_reward
                    }
                )
            
            # Curriculum Learning: Update performance and check for level advancement
            if self.use_curriculum_learning and self.curriculum_scheduler:
                self.curriculum_scheduler.update_performance(
                    success=bool(terminated),
                    reward=episode_reward,
                    episode_length=episode_length
                )
                
                # Check if should advance to next level
                if self.curriculum_scheduler.should_advance():
                    old_level = self.curriculum_scheduler.get_current_level()
                    self.curriculum_scheduler.advance_level()
                    new_level = self.curriculum_scheduler.get_current_level()
                    logger.info(f"Curriculum LEVEL UP! {old_level.name} -> {new_level.name}")
            
            # Logging với advanced metrics
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(list(self.episode_rewards)[-10:])
                mean_length = np.mean(list(self.episode_lengths)[-10:])
                mean_loss = np.mean(list(self.training_losses)[-100:]) if self.training_losses else 0.0
                mean_epsilon = np.mean(list(self.epsilon_history)[-10:])
                mean_episode_time = np.mean(list(self.episode_times)[-10:])
                mean_coverage = np.mean(list(self.episode_coverage)[-10:]) if self.episode_coverage else 0.0
                mean_success = np.mean(list(self.episode_success)[-10:]) if self.episode_success else 0.0
                mean_reward_per_step = np.mean(list(self.reward_per_step)[-10:]) if self.reward_per_step else 0.0
                
                logger.info(
                    f"Episode {episode + 1}/{max_episodes} | "
                    f"Reward: {episode_reward:.1f} (avg: {mean_reward:.1f}) | "
                    f"Length: {episode_length} (avg: {mean_length:.1f}) | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Coverage: {coverage:.2%} (avg: {mean_coverage:.2%}) | "
                    f"Success: {success} (avg: {mean_success:.2%}) | "
                    f"Reward/Step: {reward_per_step:.2f} (avg: {mean_reward_per_step:.2f}) | "
                    f"Time: {mean_episode_time:.2f}s"
                )
                
                # Tensorboard logging - organized by category
                if self.writer:
                    # Training metrics
                    self.writer.add_scalar('1_Training/Reward', episode_reward, episode)
                    self.writer.add_scalar('1_Training/Mean_Reward_10ep', mean_reward, episode)
                    self.writer.add_scalar('1_Training/Reward_Per_Step', reward_per_step, episode)
                    self.writer.add_scalar('1_Training/Mean_Reward_Per_Step_10ep', mean_reward_per_step, episode)
                    self.writer.add_scalar('1_Training/Episode_Length', episode_length, episode)
                    self.writer.add_scalar('1_Training/Mean_Length_10ep', mean_length, episode)
                    self.writer.add_scalar('1_Training/Loss', mean_loss, episode)
                    self.writer.add_scalar('1_Training/Epsilon', agent.epsilon, episode)
                    
                    # Coverage metrics
                    self.writer.add_scalar('2_Coverage/Node_Coverage', coverage, episode)
                    self.writer.add_scalar('2_Coverage/Mean_Coverage_10ep', mean_coverage, episode)
                    
                    # Success metrics
                    self.writer.add_scalar('3_Success/Success_Rate', success, episode)
                    self.writer.add_scalar('3_Success/Mean_Success_Rate_10ep', mean_success, episode)
                    
                    # Performance metrics
                    self.writer.add_scalar('4_Performance/Episode_Time', episode_time, episode)
                    self.writer.add_scalar('4_Performance/Mean_Time_10ep', mean_episode_time, episode)
                    self.writer.add_scalar('4_Performance/Memory_Usage_MB', self._get_memory_usage(), episode)
                    
                    # Agent metrics
                    if len(episode_losses) > 0:
                        agent_stats = agent.get_training_stats()
                        self.writer.add_scalar('5_Agent/Q_Value', agent_stats.get('avg_q_value', 0), episode)
                        self.writer.add_scalar('5_Agent/Buffer_Size', agent_stats.get('buffer_size', 0), episode)
                        self.writer.add_scalar('5_Agent/Learning_Rate', agent_stats.get('learning_rate', 0), episode)
            
            # Evaluation với early stopping
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward, eval_metrics = self.evaluate_advanced(
                    agent, env, num_episodes=self.eval_episodes
                )
                
                logger.info(
                    f"Evaluation at episode {episode + 1}: "
                    f"Mean reward = {eval_reward:.2f} | "
                    f"Success rate = {eval_metrics['success_rate']:.2f} | "
                    f"Avg hops = {eval_metrics['mean_hops']:.1f}"
                )
                
                if self.writer:
                    # Evaluation metrics - organized
                    self.writer.add_scalar('6_Evaluation/Mean_Reward', eval_reward, episode)
                    self.writer.add_scalar('6_Evaluation/Success_Rate', eval_metrics['success_rate'], episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Hops', eval_metrics['mean_hops'], episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Latency', eval_metrics['mean_latency'], episode)
                    self.writer.add_scalar('6_Evaluation/Std_Reward', eval_metrics['std_reward'], episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Length', eval_metrics['mean_length'], episode)
                    
                    # Reward per episode ratio
                    if eval_metrics['mean_length'] > 0:
                        eval_reward_per_step = eval_reward / eval_metrics['mean_length']
                        self.writer.add_scalar('6_Evaluation/Reward_Per_Step', eval_reward_per_step, episode)
                
                # Early stopping và save best model
                if eval_reward > self.best_mean_reward:
                    self.best_mean_reward = eval_reward
                    self.early_stopping_counter = 0
                    
                    best_model_path = self.best_model_path / 'best_model.pt'
                    agent.save(str(best_model_path))
                    
                    # Save training state
                    self._save_training_state(agent, episode)
                    
                    logger.info(f"🚀 New best model: reward = {eval_reward:.2f}")
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at episode {episode + 1}")
                        break
            
            # Checkpoint với optimizations
            if (episode + 1) % self.save_frequency == 0:
                checkpoint_path = self.checkpoint_path / f'checkpoint_ep{episode + 1}.pt'
                agent.save(str(checkpoint_path))
                
                # Save training metrics
                self._save_training_metrics(episode)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final save và cleanup
        training_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        final_model_path = self.model_path / 'final_model.pt'
        agent.save(str(final_model_path))
        
        # Save final metrics
        self._save_final_metrics()
        
        logger.info(f"Final model saved to {final_model_path}")
        
        if self.writer:
            self.writer.close()
        
        # Show final dashboard summary
        if self.dashboard_enabled:
            final_success_rate = np.mean(list(self.episode_success)) if self.episode_success else 0
            final_metrics = {
                'total_episodes': self.episode + 1,
                'total_steps': self.total_steps,
                'best_mean_reward': self.best_mean_reward,
                'final_success_rate': f"{final_success_rate:.1%}",
                'training_time': f"{training_time:.1f}s"
            }
            self.dashboard.show_final_summary(final_metrics)
        
        return agent
    
    def evaluate_advanced(
        self,
        agent: DuelingDQNAgent,
        env: RoutingEnvironment,
        num_episodes: int = 10
    ) -> Tuple[float, Dict]:
        """Advanced evaluation với comprehensive metrics"""
        agent.eval()
        eval_rewards = []
        eval_lengths = []
        eval_hops = []
        eval_latencies = []
        success_count = 0
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            # Extract additional metrics từ info
            if 'hops' in info:
                eval_hops.append(info['hops'])
            if 'total_latency' in info:
                eval_latencies.append(info['total_latency'])
            if terminated:  # Success
                success_count += 1
        
        agent.train_mode()
        
        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': success_count / num_episodes,
            'mean_hops': np.mean(eval_hops) if eval_hops else 0,
            'mean_latency': np.mean(eval_latencies) if eval_latencies else 0,
            'rewards': eval_rewards,
            'lengths': eval_lengths
        }
        
        return metrics['mean_reward'], metrics
    
    def evaluate(
        self,
        agent: DuelingDQNAgent,
        env: RoutingEnvironment,
        num_episodes: int = 10
    ) -> float:
        """Giữ nguyên interface cũ"""
        mean_reward, _ = self.evaluate_advanced(agent, env, num_episodes)
        return mean_reward
    
    def train_from_database(
        self,
        num_episodes: Optional[int] = None
    ):
        """Optimized training từ database"""
        from models.database import db
        
        try:
            # Get nodes và terminals từ database
            nodes_collection = db.get_collection('nodes')
            terminals_collection = db.get_collection('terminals')
            
            nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
            terminals = list(terminals_collection.find({}, {'_id': 0}))
            
            if len(nodes) == 0:
                raise ValueError("No operational nodes in database")
            if len(terminals) < 2:
                raise ValueError("Need at least 2 terminals in database")
            
            # Count total nodes for logging
            total_nodes_count = nodes_collection.count_documents({})
            operational_nodes_count = len(nodes)
            
            logger.info(
                f"Loaded {operational_nodes_count} operational nodes "
                f"(out of {total_nodes_count} total) and {len(terminals)} terminals from database"
            )
            
            # Pre-process nodes để tăng performance
            processed_nodes = self._preprocess_nodes(nodes)
            
            # Train
            return self.train(nodes=processed_nodes, terminals=terminals, episodes=num_episodes)
            
        except Exception as e:
            logger.error(f"Database training failed: {e}", exc_info=True)
            raise
    
    def load_and_evaluate(
        self,
        model_path: str,
        nodes: List[Dict],
        terminals: List[Dict],
        num_episodes: int = 10
    ) -> Dict:
        """Optimized evaluation với comprehensive metrics"""
        # Initialize environment
        env = RoutingEnvironment(
            nodes=nodes,
            terminals=terminals,
            config=self.config,
            max_steps=self.max_steps_per_episode
        )
        
        # Initialize state builder
        state_builder = RoutingStateBuilder(self.config)
        state_dim = state_builder.state_dimension
        action_dim = env.action_space.n
        
        # Load agent
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.config
        )
        agent.load(model_path)
        agent.eval()
        
        # Evaluate với advanced metrics
        _, metrics = self.evaluate_advanced(agent, env, num_episodes)
        
        return metrics
    
    def _preprocess_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """Pre-process nodes để tăng training performance"""
        processed_nodes = []
        
        for node in nodes:
            # Đảm bảo các fields cần thiết tồn tại
            processed_node = node.copy()
            
            # Set default values cho missing fields
            if 'resourceUtilization' not in processed_node:
                processed_node['resourceUtilization'] = 0
            if 'packetLossRate' not in processed_node:
                processed_node['packetLossRate'] = 0
            if 'nodeProcessingDelayMs' not in processed_node:
                processed_node['nodeProcessingDelayMs'] = 5
            if 'batteryChargePercent' not in processed_node:
                processed_node['batteryChargePercent'] = 100
            if 'currentPacketCount' not in processed_node:
                processed_node['currentPacketCount'] = 0
            if 'packetBufferCapacity' not in processed_node:
                processed_node['packetBufferCapacity'] = 1000
            if 'communication' not in processed_node:
                processed_node['communication'] = {'bandwidth': 100, 'maxRangeKm': 2000}
                
            processed_nodes.append(processed_node)
        
        return processed_nodes
    
    def _save_training_state(self, agent: DuelingDQNAgent, episode: int):
        """Save training state để resume sau này"""
        state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_mean_reward': self.best_mean_reward,
            'early_stopping_counter': self.early_stopping_counter,
            'agent_state': agent.get_state(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'training_losses': list(self.training_losses)
        }
        
        state_path = self.best_model_path / 'training_state.pt'
        torch.save(state, str(state_path))
    
    def _save_training_metrics(self, episode: int):
        """Save training metrics để analysis"""
        metrics = {
            'episode': episode,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'training_losses': list(self.training_losses),
            'epsilon_history': list(self.epsilon_history),
            'best_mean_reward': self.best_mean_reward
        }
        
        metrics_path = self.checkpoint_path / f'metrics_ep{episode}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_final_metrics(self):
        """Save final training metrics"""
        final_metrics = {
            'total_episodes': self.episode,
            'total_steps': self.total_steps,
            'best_mean_reward': self.best_mean_reward,
            'final_timestamp': datetime.now().isoformat(),
            'final_epsilon': self.epsilon_history[-1] if self.epsilon_history else 0,
            'mean_reward_last_100': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
            'mean_length_last_100': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0
        }
        
        final_metrics_path = self.model_path / 'final_metrics.json'
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024