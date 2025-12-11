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
        
        # Generate expert demonstrations nếu dùng imitation learning
        if self.use_imitation and len(self.imitation.expert_demos) == 0:
            logger.info("Generating expert demonstrations...")
            self._generate_expert_demos(terminals, nodes, num_demos=50)
        
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
                
                # Tensorboard
                if self.writer:
                    self.writer.add_scalar('train/reward', episode_reward, episode)
                    self.writer.add_scalar('train/mean_reward', mean_reward, episode)
                    self.writer.add_scalar('train/loss', mean_loss, episode)
                    
                    if self.use_curriculum:
                        self.writer.add_scalar('curriculum/level', stats['current_level'], episode)
                        self.writer.add_scalar('curriculum/difficulty', stats['difficulty'], episode)
                    
                    if self.use_imitation:
                        self.writer.add_scalar('imitation/expert_ratio', self.imitation.mixing_ratio, episode)
            
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
                    self.writer.add_scalar('eval/mean_reward', eval_reward, episode)
                    self.writer.add_scalar('eval/success_rate', eval_metrics['success_rate'], episode)
                
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
        num_demos: int = 50
    ):
        """Generate expert demonstrations"""
        logger.info(f"Generating {num_demos} expert demonstrations...")
        
        demo_count = 0
        attempts = 0
        max_attempts = num_demos * 3
        
        while demo_count < num_demos and attempts < max_attempts:
            attempts += 1
            
            if len(terminals) < 2:
                break
            
            indices = np.random.choice(len(terminals), size=2, replace=False)
            source = terminals[indices[0]]
            dest = terminals[indices[1]]
            
            # Generate Dijkstra demonstration
            demo = self.imitation.generate_expert_demonstration(
                source, dest, nodes, algorithm='dijkstra'
            )
            
            if demo:
                self.imitation.add_demonstration(demo)
                demo_count += 1
        
        logger.info(f"Generated {demo_count} expert demonstrations")

