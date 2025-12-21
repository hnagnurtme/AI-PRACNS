"""
Enhanced Trainer với Curriculum Learning, Imitation Learning, và Multi-objective Optimization
"""
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time
import random

from training.trainer import RoutingTrainer
from training.curriculum_learning import CurriculumScheduler
from training.imitation_learning import ImitationLearning
from training.multi_objective import MultiObjectiveOptimizer
from agent.dueling_dqn import DuelingDQNAgent
from environment.routing_env import RoutingEnvironment
from environment.state_builder import RoutingStateBuilder

logger = logging.getLogger(__name__)


class EnhancedRoutingTrainer(RoutingTrainer):
    """
    Enhanced trainer với:
    - Curriculum Learning
    - Imitation Learning
    - Multi-objective Optimization
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Initialize advanced modules
        self.curriculum = CurriculumScheduler(config)
        self.imitation = ImitationLearning(config)
        self.multi_objective = MultiObjectiveOptimizer(config)
        
        # Training state
        self.use_curriculum = config.get('curriculum', {}).get('enabled', True)
        self.use_imitation = config.get('imitation_learning', {}).get('enabled', True)
        self.use_multi_objective = config.get('multi_objective', {}).get('enabled', True)
        
        logger.info("Enhanced Routing Trainer initialized")
        logger.info(f"  Curriculum Learning: {self.use_curriculum}")
        logger.info(f"  Imitation Learning: {self.use_imitation}")
        logger.info(f"  Multi-objective: {self.use_multi_objective}")
    
    def train(
        self,
        nodes: List[Dict],
        terminals: List[Dict],
        episodes: Optional[int] = None
    ):
        """Enhanced training loop với advanced features"""
        max_episodes = episodes or self.max_episodes
        self.training_start_time = time.time()
        
        # Store original nodes for curriculum filtering
        original_nodes = nodes.copy()
        
        # Filter nodes theo curriculum level
        if self.use_curriculum:
            filtered_nodes = self.curriculum.filter_nodes(nodes)
            logger.info(f"Curriculum Level: {self.curriculum.get_current_level().name}")
            logger.info(f"Filtered to {len(filtered_nodes)} nodes from {len(nodes)}")
            nodes = filtered_nodes
        
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
        
        # Initialize agent
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.config
        )
        
        logger.info(f"Starting enhanced training: {max_episodes} episodes")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        
        if self.use_imitation:
            # Clear old demos if state dimension changed (e.g., max_nodes changed)
            if len(self.imitation.expert_demos) > 0:
                # Check if demo state dimension matches current state dimension
                first_demo = self.imitation.expert_demos[0]
                if len(first_demo.states) > 0:
                    demo_state_dim = len(first_demo.states[0])
                    if demo_state_dim != state_dim:
                        logger.info(f"State dimension changed ({demo_state_dim} → {state_dim}), clearing old demonstrations")
                        self.imitation.expert_demos.clear()
            
            if len(self.imitation.expert_demos) == 0:
                logger.info("Generating expert demonstrations...")
                imitation_config = self.config.get('imitation_learning', {})
                num_demos = imitation_config.get('num_demos', 500)
                self.imitation.generate_comprehensive_demos(terminals, nodes, num_demos=num_demos)
        
        # Training loop
        for episode in range(max_episodes):
            self.episode = episode
            episode_start_time = time.time()
            
            # Select terminals theo curriculum
            source_terminal, dest_terminal = self._select_terminals(
                terminals, nodes
            )
            
            if not source_terminal or not dest_terminal:
                continue
            
            # Reset environment với selected terminals
            state, info = env.reset(
                options={
                    'source_terminal_id': source_terminal.get('terminalId'),
                    'dest_terminal_id': dest_terminal.get('terminalId')
                }
            )
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            episode_losses = []
            episode_path = []
            
            # Episode loop
            while not done:
                # Action selection với imitation learning
                if self.use_imitation and random.random() < self.imitation.mixing_ratio:
                    # Use expert action
                    filtered_nodes = state_builder._smart_node_filtering(
                        nodes, source_terminal, dest_terminal,
                        env.current_node, list(env.visited_nodes)
                    )
                    expert_action = self.imitation.sample_expert_action(
                        state, filtered_nodes, env.current_node, dest_terminal
                    )
                    if expert_action is not None:
                        action = expert_action
                    else:
                        action = agent.select_action(state, deterministic=False)
                else:
                    # Use agent action
                    action = agent.select_action(state, deterministic=False)
                
                # Step environment
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Multi-objective reward adjustment
                if self.use_multi_objective and done:
                    path = env.path.copy()
                    if path:
                        mo_reward, objectives = self.multi_objective.compute_reward_with_objectives(
                            path, nodes
                        )
                        # Combine với original reward
                        reward = 0.7 * reward + 0.3 * mo_reward
                
                # Store experience
                agent.replay_buffer.push(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                # Train agent
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
                
                # Update target network
                if agent.total_steps % self.target_update_frequency == 0:
                    agent.update_target_network()
                
                max_training_steps = max_episodes * self.max_steps_per_episode
                agent.update_epsilon(agent.total_steps, max_training_steps)
            
            # Update curriculum
            if self.use_curriculum:
                success = terminated
                self.curriculum.update_performance(
                    success=success,
                    reward=episode_reward,
                    episode_length=episode_length
                )
                
                if self.curriculum.should_advance():
                    self.curriculum.advance_level()
                    # Re-filter nodes for new level
                    filtered_nodes = self.curriculum.filter_nodes(original_nodes)
                    logger.info(f"Advanced to level {self.curriculum.current_level}, filtered to {len(filtered_nodes)} nodes")
                    nodes = filtered_nodes
                    # Update environment với nodes mới
                    env = RoutingEnvironment(
                        nodes=nodes,
                        terminals=terminals,
                        config=self.config,
                        max_steps=self.max_steps_per_episode
                    )
            
            # Update imitation learning (DAGGER)
            if self.use_imitation:
                success_rate = len([r for r in self.episode_rewards if r > 0]) / max(len(self.episode_rewards), 1)
                self.imitation.update_dagger(success_rate)
            
            # Calculate coverage and success
            visited_nodes = len(env.visited_nodes) if hasattr(env, 'visited_nodes') else 0
            total_nodes = len(nodes)
            coverage = visited_nodes / total_nodes if total_nodes > 0 else 0.0
            if not hasattr(self, 'episode_coverage'):
                self.episode_coverage = deque(maxlen=100)
                self.episode_success = deque(maxlen=100)
                self.reward_per_step = deque(maxlen=100)
            
            self.episode_coverage.append(coverage)
            success = 1 if terminated else 0
            self.episode_success.append(success)
            reward_per_step = episode_reward / episode_length if episode_length > 0 else 0.0
            self.reward_per_step.append(reward_per_step)
            
            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(agent.epsilon)
            
            episode_time = time.time() - episode_start_time
            self.episode_times.append(episode_time)
            
            # Logging
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(list(self.episode_rewards)[-10:])
                mean_length = np.mean(list(self.episode_lengths)[-10:])
                mean_loss = np.mean(list(self.training_losses)[-100:]) if self.training_losses else 0.0
                
                log_msg = (
                    f"Episode {episode + 1}/{max_episodes} | "
                    f"Reward: {episode_reward:.1f} (avg: {mean_reward:.1f}) | "
                    f"Length: {episode_length} (avg: {mean_length:.1f}) | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )
                
                if self.use_curriculum:
                    stats = self.curriculum.get_stats()
                    log_msg += f" | Level: {stats['level_name']} ({stats['current_level']}/{stats['total_levels']-1})"
                
                if self.use_imitation:
                    log_msg += f" | Expert: {self.imitation.mixing_ratio:.2f}"
                
                logger.info(log_msg)
                
                # Calculate additional metrics
                mean_coverage = np.mean(list(self.episode_coverage)[-10:]) if hasattr(self, 'episode_coverage') and self.episode_coverage else 0.0
                mean_success = np.mean(list(self.episode_success)[-10:]) if hasattr(self, 'episode_success') and self.episode_success else 0.0
                mean_reward_per_step = np.mean(list(self.reward_per_step)[-10:]) if hasattr(self, 'reward_per_step') and self.reward_per_step else 0.0
                
                # Tensorboard - organized by category
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
                    
                    # Enhanced features
                    if self.use_curriculum:
                        self.writer.add_scalar('7_Enhanced/Curriculum_Level', stats['current_level'], episode)
                        self.writer.add_scalar('7_Enhanced/Curriculum_Difficulty', stats['difficulty'], episode)
                    
                    if self.use_imitation:
                        self.writer.add_scalar('7_Enhanced/Imitation_Expert_Ratio', self.imitation.mixing_ratio, episode)
            
            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward, eval_metrics = self.evaluate_advanced(
                    agent, env, num_episodes=self.eval_episodes
                )
                
                logger.info(
                    f"Evaluation at episode {episode + 1}: "
                    f"Mean reward = {eval_reward:.2f} | "
                    f"Success rate = {eval_metrics['success_rate']:.2f}"
                )
                
                if self.writer:
                    # Evaluation metrics - organized
                    self.writer.add_scalar('6_Evaluation/Mean_Reward', eval_reward, episode)
                    self.writer.add_scalar('6_Evaluation/Success_Rate', eval_metrics['success_rate'], episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Hops', eval_metrics.get('mean_hops', 0), episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Latency', eval_metrics.get('mean_latency', 0), episode)
                    self.writer.add_scalar('6_Evaluation/Std_Reward', eval_metrics.get('std_reward', 0), episode)
                    self.writer.add_scalar('6_Evaluation/Mean_Length', eval_metrics.get('mean_length', 0), episode)
                    
                    # Reward per episode ratio
                    if eval_metrics.get('mean_length', 0) > 0:
                        eval_reward_per_step = eval_reward / eval_metrics['mean_length']
                        self.writer.add_scalar('6_Evaluation/Reward_Per_Step', eval_reward_per_step, episode)
                
                # Early stopping và save best model
                if eval_reward > self.best_mean_reward:
                    self.best_mean_reward = eval_reward
                    self.early_stopping_counter = 0
                    
                    best_model_path = self.best_model_path / 'best_model.pt'
                    agent.save(str(best_model_path))
                    logger.info(f"🚀 New best model: reward = {eval_reward:.2f}")
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at episode {episode + 1}")
                        break
            
            # Checkpoint
            if (episode + 1) % self.save_frequency == 0:
                checkpoint_path = self.checkpoint_path / f'checkpoint_ep{episode + 1}.pt'
                agent.save(str(checkpoint_path))
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final save
        training_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        final_model_path = self.model_path / 'final_model.pt'
        agent.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        if self.writer:
            self.writer.close()
        
        return agent
    
    def _select_terminals(
        self,
        terminals: List[Dict],
        nodes: List[Dict]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Select terminals theo curriculum level"""
        if len(terminals) < 2:
            return None, None
        
        if not self.use_curriculum:
            # Random selection
            indices = np.random.choice(len(terminals), size=2, replace=False)
            return terminals[indices[0]], terminals[indices[1]]
        
        # Filter terminals theo curriculum
        level = self.curriculum.get_current_level()
        valid_pairs = []
        
        for i, source in enumerate(terminals):
            for j, dest in enumerate(terminals):
                if i == j:
                    continue
                
                is_valid, reason = self.curriculum.filter_scenario(source, dest, nodes)
                if is_valid:
                    valid_pairs.append((source, dest))
        
        if valid_pairs:
            return random.choice(valid_pairs)
        else:
            # Fallback to random
            indices = np.random.choice(len(terminals), size=2, replace=False)
            return terminals[indices[0]], terminals[indices[1]]
    
    def _generate_expert_demos(
        self,
        terminals: List[Dict],
        nodes: List[Dict],
        num_demos: int = 500
    ):
        """Generate expert demonstrations using comprehensive method"""
        self.imitation.generate_comprehensive_demos(terminals, nodes, num_demos=num_demos)

