# training/training_manager.py
import time
import random
import torch
import numpy as np
from typing import List, Tuple
from datetime import datetime

class TrainingManager:
    """Quản lý quá trình training của DRL agent"""
    
    def __init__(self, config, mongo_manager, resume_from_checkpoint=None):
        self.config = config
        self.mongo_manager = mongo_manager
        
        # Resume settings
        self.resume_checkpoint = resume_from_checkpoint
        
        # Khởi tạo components
        self._setup_components()
        
        # Training state
        self.episode = 0
        self.global_step = 0
        self.best_avg_reward = -float('inf')
        
        # CRITICAL FIXES:
        # 1. Increase patience for early stopping
        self.patience = 3000  # Was 500, now 2000
        self.episodes_without_improvement = 0
        
        # 2. Better evaluation frequency
        self.evaluation_frequency = 500  # Evaluate every 200 episodes instead of 100
        
        # 3. Minimum episodes before early stopping
        self.min_episodes_before_stopping = 2000

        # Load checkpoint nếu có
        if self.resume_checkpoint:
            self._load_checkpoint()
        
        # Real-time data refresh settings
        self.last_data_refresh = 0
        self.data_refresh_interval = 60
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []

    def _setup_components(self):
        """Khởi tạo tất cả components cần thiết"""
        from data.data_pipeline import RealTimeDataPipeline
        from env.link_metrics_calculator import LinkMetricsCalculator
        from env.reward_calculator import RewardCalculator
        from env.state_processor import StateProcessor
        from env.action_mapper import ActionMapper
        from agents.dpn_agent import DqnAgent
        from agents.replay_buffer import InMemoryReplayBuffer
        from simulator.network_simulator import NetworkSimulator
        
        print("🔧 Setting up components...")
        
        try:
            # 1. Data Pipeline
            print("   📊 Initializing Data Pipeline...")
            self.data_pipeline = RealTimeDataPipeline(self.mongo_manager, 60)
            print("   ✅ Data Pipeline initialized")
            
            # 2. Environment components
            print("   🌐 Initializing Environment Components...")
            self.link_calculator = LinkMetricsCalculator()
            print("   ✅ Link Metrics Calculator initialized")
            
            self.reward_calculator = RewardCalculator()
            print("   ✅ Reward Calculator initialized")
            
            self.state_processor = StateProcessor(max_neighbors=10)
            state_size = self.state_processor.get_state_size()
            print(f"   ✅ State Processor initialized: state_size={state_size}")
            
            self.action_mapper = ActionMapper(self.mongo_manager)
            action_size = self.action_mapper.get_action_size()
            print(f"   ✅ Action Mapper initialized: action_size={action_size}")
            
            # 3. DQN Agent
            print("   🤖 Initializing DQN Agent...")
            self.agent = DqnAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma
            )
            self.agent.action_mapper = self.action_mapper
            print("   ✅ DQN Agent initialized")
            
            # 4. Replay Buffer
            print("   💾 Initializing Replay Buffer...")
            self.buffer = InMemoryReplayBuffer(capacity=100000)
            print(f"   ✅ Replay Buffer initialized: capacity={self.buffer.get_capacity()}")
            
            # 5. Simulator
            print("   🎮 Initializing Network Simulator...")
            self.simulator = NetworkSimulator(
                agent=self.agent,
                buffer=self.buffer,
                data_pipeline=self.data_pipeline,
                link_calculator=self.link_calculator,
                reward_calculator=self.reward_calculator,
                state_processor=self.state_processor
            )
            print("   ✅ Network Simulator initialized")
            
            print("🎉 All components setup successfully!")
            
        except Exception as e:
            print(f"❌ Error during component setup: {e}")
            raise

    def _load_checkpoint(self):
        """Load checkpoint để resume training"""
        try:
            print(f"📂 Loading checkpoint: {self.resume_checkpoint}")
            
            checkpoint = torch.load(self.resume_checkpoint, map_location='cpu')
            
            # Restore model state
            self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            self.episode = checkpoint.get('episode', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_avg_reward = checkpoint.get('avg_reward', -float('inf'))
            self.agent.epsilon = checkpoint.get('epsilon', 0.1)
            
            # Load metrics if available
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_losses = checkpoint.get('episode_losses', [])
            
            print(f"✅ Resumed from Episode {self.episode}")
            print(f"   Best Reward: {self.best_avg_reward:.2f}")
            print(f"   Epsilon: {self.agent.epsilon:.3f}")
            print(f"   Global Step: {self.global_step}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("🔄 Starting fresh training...")

    def start_training(self):
        """Bắt đầu training process"""
        print(f"🚀 Starting DRL Training Process...")
        print(f"   Episodes: {self.episode} → {self.config.total_episodes}")
        print(f"   Learning Rate: {self.config.learning_rate}")
        print(f"   Batch Size: {self.config.batch_size}")
        
        try:
            # 1. Khởi động Data Pipeline
            self.data_pipeline.start()
            
            # 2. Validation check
            if not self._validate_network():
                print("❌ Network validation failed!")
                return
            
            # 3. Warm-up phase (chỉ khi bắt đầu fresh)
            if self.episode == 0:
                self._warm_up_phase()
            
            # 4. Main training loop
            self._main_training_loop()
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user.")
            self._save_current_progress()
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            self._save_current_progress()
        finally:
            self.data_pipeline.stop()
            print("🏁 Training process ended.")

    def _validate_network(self) -> bool:
        """Validate network có đủ nodes để training"""
        try:
            snapshot = self.data_pipeline.get_current_snapshot()
            nodes = snapshot.get('nodes', {})
            
            if len(nodes) < 3:
                print(f"❌ Not enough nodes for training: {len(nodes)}")
                return False
            
            # Check node types
            node_types = {}
            for node_data in nodes.values():
                node_type = node_data.get('nodeType', 'UNKNOWN')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"📊 Network Validation:")
            print(f"   Total Nodes: {len(nodes)}")
            for node_type, count in node_types.items():
                print(f"   {node_type}: {count}")
            
            # Ensure có ít nhất 2 loại nodes khác nhau
            if len(node_types) < 2:
                print("⚠️ Warning: Only one type of node found")
            
            return True
            
        except Exception as e:
            print(f"❌ Network validation error: {e}")
            return False

    def _warm_up_phase(self):
        """Giai đoạn khởi động với better exploration"""
        print(f"🔥 Warm-up phase: {self.config.warmup_steps} steps...")
        
        warm_up_rewards = []
        
        for step in range(self.config.warmup_steps):
            try:
                # Get fresh nodes
                snapshot = self.data_pipeline.get_current_snapshot()
                available_nodes = list(snapshot.get('nodes', {}).keys())
                
                if len(available_nodes) < 2:
                    print(f"⚠️ Not enough nodes at step {step}")
                    continue
                
                source, dest = random.sample(available_nodes, 2)
                
                # High exploration during warm-up
                original_epsilon = self.agent.epsilon
                self.agent.epsilon = 0.9
                
                # Run simulation
                result = self.simulator.simulation_one_step(source, dest)
                warm_up_rewards.append(result['reward'])
                self.global_step += 1
                
                # Restore epsilon
                self.agent.epsilon = original_epsilon
                
                if (step + 1) % 200 == 0:
                    avg_reward = np.mean(warm_up_rewards[-200:])
                    print(f"   Step {step + 1}/{self.config.warmup_steps} | "
                          f"Buffer: {self.buffer.get_size()} | Avg Reward: {avg_reward:.2f}")
                    
            except Exception as e:
                print(f"❌ Error in warm-up step {step}: {e}")
                continue
        
        final_avg = np.mean(warm_up_rewards) if warm_up_rewards else 0
        print(f"✅ Warm-up completed | Buffer: {self.buffer.get_size()} | Avg Reward: {final_avg:.2f}")

    def _main_training_loop(self):
        """Vòng lặp training chính với improvements"""
        print(f"\n🚀 Main training loop: Episodes {self.episode} → {self.config.total_episodes}")
        
        start_episode = self.episode
        
        for episode in range(start_episode, self.config.total_episodes):
            self.episode = episode
            
            try:
                # Force refresh data định kỳ
                current_time = time.time()
                if (current_time - self.last_data_refresh > self.data_refresh_interval) or (episode % 50 == 0):
                    self._force_refresh_data()
                    self.last_data_refresh = current_time
                
                # Epsilon adjustment
                self._adjust_epsilon(episode)
                
                # Run training episode với better routing
                episode_reward = self._run_better_training_episode()
                self.episode_rewards.append(episode_reward)
                
                # Training with batch
                training_loss = self._train_with_batch()
                self.episode_losses.append(training_loss)
                
                # More frequent logging but less frequent evaluation
                if episode % 50 == 0:
                    self._log_process(episode, episode_reward, training_loss)
                
                # Less frequent evaluation
                if episode % self.evaluation_frequency == 0 and episode > 0:
                    should_stop = self._evaluate_and_save(episode)
                    if should_stop:
                        print("🛑 Early stopping triggered!")
                        break
                
                # Less frequent checkpointing
                if episode % 1000 == 0 and episode > 0:
                    self._save_checkpoint(episode, "periodic")
                    
            except Exception as e:
                print(f"❌ Error in episode {episode}: {e}")
                continue
        
        # Final save
        self._save_checkpoint(self.episode, "final")
        print(f"🏁 Training completed after {self.episode} episodes")

    def _run_better_training_episode(self) -> float:
        """Improved training episode với better loop prevention"""
        try:
            snapshot = self.data_pipeline.get_current_snapshot()
            nodes_data = snapshot.get('nodes', {})
            available_nodes = list(nodes_data.keys())
            
            if len(available_nodes) < 2:
                return -20.0
            
            # Chọn source và destination cách xa nhau để tránh routes quá ngắn
            source, dest = self._select_distant_nodes(nodes_data, available_nodes)
            
            episode_reward = 0.0
            current_node = source
            path = [source]
            max_hops = 8  # Giảm max hops để tránh loops dài
            
            print(f"🚀 Starting episode: {source} -> {dest} (max {max_hops} hops)")
            
            for hop in range(max_hops):
                if current_node == dest:
                    # SUCCESS - High reward
                    success_bonus = 150.0 + (50.0 * (1.0 - hop / max_hops))
                    episode_reward += success_bonus
                    print(f"🎉 Episode SUCCESS in {hop} hops! Bonus: {success_bonus:.1f}")
                    break
                
                try:
                    result = self.simulator.simulation_one_step(
                        current_node, 
                        dest, 
                        path_history=path,
                        current_step=hop,
                        max_steps=max_hops
                    )
                    next_node = result['action_taken']
                    reward = result['reward']
                    
                    episode_reward += reward
                    self.global_step += 1
                    
                    # Update current node
                    if next_node not in nodes_data:
                        episode_reward -= 10.0
                        print(f"❌ Invalid node: {next_node}")
                        break
                    
                    # Loop detection và penalty
                    if next_node in path:
                        episode_reward -= 25.0  # Tăng loop penalty
                        print(f"🔄 Loop detected: {next_node} in path, breaking")
                        break
                    
                    path.append(next_node)
                    current_node = next_node
                    
                    # Early success
                    if result.get('reached_destination', False):
                        success_bonus = 200.0 * (1.0 - (hop + 1) / max_hops)
                        episode_reward += success_bonus
                        print(f"🎉 Early success at hop {hop}! Bonus: {success_bonus:.1f}")
                        break
                        
                except Exception as e:
                    print(f"❌ Error in training step {hop}: {e}")
                    episode_reward -= 10.0
                    break
            
            # Final evaluation
            if current_node != dest:
                episode_reward -= 30.0  # Failure penalty
                print(f"💥 Episode FAILED: reached {current_node} instead of {dest}")
            else:
                print(f"✅ Episode COMPLETED: {source} -> {dest} in {len(path)-1} hops")
            
            print(f"📊 Episode reward: {episode_reward:.1f}, Path: {' -> '.join(path)}")
            return episode_reward
            
        except Exception as e:
            print(f"❌ Error in training episode: {e}")
            return -20.0

    def _select_distant_nodes(self, nodes_data: dict, available_nodes: List[str]) -> Tuple[str, str]:
        """Chọn source và destination cách xa nhau"""
        if len(available_nodes) < 4:
            return random.sample(available_nodes, 2)
        
        # Ưu tiên chọn nodes khác loại và khác vùng địa lý
        ground_stations = [nid for nid, ndata in nodes_data.items() 
                        if ndata.get('nodeType') == 'GROUND_STATION']
        sea_stations = [nid for nid, ndata in nodes_data.items() 
                    if ndata.get('nodeType') == 'SEA_STATION']
        satellites = [nid for nid, ndata in nodes_data.items() 
                    if 'SATELLITE' in ndata.get('nodeType', '')]
        
        # Tạo các cặp source-dest có khả năng cao
        pairs = []
        
        if ground_stations and sea_stations:
            pairs.append((random.choice(ground_stations), random.choice(sea_stations)))
        if ground_stations and satellites:
            pairs.append((random.choice(ground_stations), random.choice(satellites)))
        if sea_stations and satellites:
            pairs.append((random.choice(sea_stations), random.choice(satellites)))
        
        if pairs:
            return random.choice(pairs)
        else:
            return random.sample(available_nodes, 2)

    def _select_curriculum_nodes(self, nodes_data: dict, episode: int) -> Tuple[str, str]:
        """Chọn nodes theo curriculum learning"""
        available_nodes = list(nodes_data.keys())
        
        # Phân loại nodes theo độ khó
        ground_stations = [nid for nid, ndata in nodes_data.items() 
                          if ndata.get('nodeType') == 'GROUND_STATION']
        sea_stations = [nid for nid, ndata in nodes_data.items() 
                       if ndata.get('nodeType') == 'SEA_STATION']
        satellites = [nid for nid, ndata in nodes_data.items() 
                     if 'SATELLITE' in ndata.get('nodeType', '')]
        
        # Curriculum phases
        if episode < 1000:  # Phase 1: Ground stations only
            if len(ground_stations) >= 2:
                return random.sample(ground_stations, 2)
        
        elif episode < 3000:  # Phase 2: Ground + Sea stations
            if ground_stations and sea_stations:
                return random.choice(ground_stations), random.choice(sea_stations)
        
        elif episode < 6000:  # Phase 3: Include satellites
            scenarios = []
            if ground_stations and sea_stations:
                scenarios.append(('ground', 'sea'))
            if sea_stations and satellites:
                scenarios.append(('sea', 'satellite'))
            if ground_stations and satellites:
                scenarios.append(('ground', 'satellite'))
            
            if scenarios:
                scenario = random.choice(scenarios)
                if scenario == ('ground', 'sea'):
                    return random.choice(ground_stations), random.choice(sea_stations)
                elif scenario == ('sea', 'satellite'):
                    return random.choice(sea_stations), random.choice(satellites)
                else:
                    return random.choice(ground_stations), random.choice(satellites)
        
        # Phase 4: Mixed complexity
        return random.sample(available_nodes, 2)

    def _get_max_hops_for_episode(self, episode: int) -> int:
        """Tăng dần độ dài route theo thời gian training"""
        if episode < 1000:
            return 4
        elif episode < 3000:
            return 6
        elif episode < 6000:
            return 8
        else:
            return 10

    def _select_training_nodes(self, nodes_data: dict) -> Tuple[str, str]:
        """Smart selection of source and destination cho training"""
        try:
            # Categorize nodes
            ground_stations = [nid for nid, ndata in nodes_data.items() 
                              if ndata.get('nodeType') == 'GROUND_STATION']
            sea_stations = [nid for nid, ndata in nodes_data.items() 
                           if ndata.get('nodeType') == 'SEA_STATION']
            satellites = [nid for nid, ndata in nodes_data.items() 
                         if 'SATELLITE' in ndata.get('nodeType', '')]
            
            # Training scenarios với varying difficulty
            scenario_weights = [0.4, 0.3, 0.2, 0.1]  # ground-sea, sea-sat, sat-ground, random
            scenario = np.random.choice(['ground-sea', 'sea-satellite', 'satellite-ground', 'random'], 
                                       p=scenario_weights)
            
            if scenario == 'ground-sea' and ground_stations and sea_stations:
                return random.choice(ground_stations), random.choice(sea_stations)
            elif scenario == 'sea-satellite' and sea_stations and satellites:
                return random.choice(sea_stations), random.choice(satellites)
            elif scenario == 'satellite-ground' and satellites and ground_stations:
                return random.choice(satellites), random.choice(ground_stations)
            else:
                # Random fallback
                available_nodes = list(nodes_data.keys())
                return random.sample(available_nodes, 2)
                
        except Exception as e:
            print(f"❌ Error in node selection: {e}")
            available_nodes = list(nodes_data.keys())
            return random.sample(available_nodes, 2)

    def _adjust_epsilon(self, episode: int):
        """More conservative epsilon decay"""
        total_episodes = self.config.total_episodes
        
        if episode < total_episodes * 0.1:  # First 20% - high exploration
            self.agent.epsilon = 0.5
        elif episode < total_episodes * 0.4:  # Next 30% - gradual reduction
            progress = (episode - total_episodes * 0.1) / (total_episodes * 0.3)
            self.agent.epsilon = 0.5 - 0.4 * progress  # 0.5 → 0.1
        elif episode < total_episodes * 0.8:  # Next 30% - moderate exploration 
            progress = (episode - total_episodes * 0.4) / (total_episodes * 0.3)
            self.agent.epsilon = 0.1 - 0.08 * progress  # 0.3 → 0.05
        else:  # Final 20% - minimal exploration
            self.agent.epsilon = max(0.01, 0.05 - 0.04 * 
                                   (episode - total_episodes * 0.8) / 
                                   (total_episodes * 0.2))

    def _train_with_batch(self) -> float:
        """Training agent với batch từ replay buffer"""
        if self.buffer.get_size() < self.config.batch_size:
            return 0.0
        
        try:
            experiences = self.buffer.sample(self.config.batch_size)
            loss = self.agent.learn(experiences)
            
            # Update target network
            if self.global_step % self.config.target_update_freq == 0:
                self.agent.update_target_network()
                
            return loss
            
        except Exception as e:
            print(f"❌ Error in batch training: {e}")
            return 0.0

    def _evaluate_and_save(self, episode: int) -> bool:
        """Modified early stopping logic"""
        try:
            print(f"📊 Evaluating at episode {episode}...")
            
            avg_reward = self._run_evaluation()
            
            # Don't trigger early stopping too early
            if episode < self.min_episodes_before_stopping:
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    self._save_model(episode, avg_reward)
                    print(f"🆕 New best model! Reward: {avg_reward:.2f}")
                return False
            
            # Check for improvement with tolerance
            improvement_threshold = 5.0  # Allow small improvements
            
            if avg_reward > self.best_avg_reward + improvement_threshold:
                improvement = avg_reward - self.best_avg_reward
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
                
                self._save_model(episode, avg_reward)
                print(f"🆕 New best model! Reward: {avg_reward:.2f} (+{improvement:.2f})")
                
            else:
                self.episodes_without_improvement += self.evaluation_frequency
                print(f"📈 No significant improvement. Best: {self.best_avg_reward:.2f}, Current: {avg_reward:.2f}")
            
            # More lenient early stopping
            if self.episodes_without_improvement >= self.patience:
                print(f"⏹️ Early stopping: No improvement for {self.patience} episodes")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            return False

    def _run_evaluation(self, num_tests: int = 10) -> float:  # FIX: Giảm số test để tránh lỗi
        """Run evaluation với better testing"""
        try:
            total_reward = 0.0
            successful_routes = 0
            tests_completed = 0
            
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.01  # FIX: Very low epsilon, not zero
            
            for test in range(num_tests):
                try:
                    snapshot = self.data_pipeline.get_current_snapshot()
                    nodes_data = snapshot.get('nodes', {})
                    available_nodes = list(nodes_data.keys())
                    
                    if len(available_nodes) < 2:
                        continue
                    
                    source, dest = random.sample(available_nodes, 2)
                    
                    # Multi-hop evaluation
                    current_node = source
                    test_reward = 0.0
                    path = [source]
                    max_hops = 6
                    
                    for hop in range(max_hops):
                        if current_node == dest:
                            test_reward += 100.0  # Success bonus
                            successful_routes += 1
                            break
                        
                        # FIX: Thêm step parameters
                        result = self.simulator.simulation_one_step(
                            current_node, 
                            dest, 
                            path_history=path,
                            current_step=hop,
                            max_steps=max_hops
                        )
                        test_reward += result['reward']
                        next_node = result['action_taken']
                        
                        # Update path và current node
                        if next_node not in nodes_data or next_node in path:
                            break
                        
                        path.append(next_node)
                        current_node = next_node
                        
                        if result.get('reached_destination', False):
                            test_reward += 50.0  # Extra success bonus
                            successful_routes += 1
                            break
                    
                    total_reward += test_reward
                    tests_completed += 1
                    
                except Exception as e:
                    print(f"❌ Error in evaluation test {test}: {e}")
                    continue
            
            self.agent.epsilon = original_epsilon
            
            if tests_completed == 0:
                return 0.0
                
            avg_reward = total_reward / tests_completed
            success_rate = successful_routes / tests_completed
            
            print(f"   ✅ Evaluation: {tests_completed}/{num_tests} tests completed")
            print(f"   📊 Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1%}")
            
            return avg_reward
            
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            return 0.0

    def _save_model(self, episode: int, avg_reward: float):
        """Save best model"""
        try:
            import os
            
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Save best model
            best_path = os.path.join(models_dir, f"best_model_ep{episode}_reward{avg_reward:.2f}.pth")
            latest_path = os.path.join(models_dir, "latest_checkpoint.pth")
            
            checkpoint = {
                'episode': episode,
                'global_step': self.global_step,
                'model_state_dict': self.agent.policy_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'avg_reward': avg_reward,
                'epsilon': self.agent.epsilon,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'episode_rewards': self.episode_rewards,
                'episode_losses': self.episode_losses,
                'config': {
                    'learning_rate': self.config.learning_rate,
                    'gamma': self.config.gamma,
                    'batch_size': self.config.batch_size
                }
            }
            
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, latest_path)
            
            print(f"💾 Model saved: {best_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def _save_checkpoint(self, episode: int, checkpoint_type: str = "periodic"):
        """Save checkpoint for resuming"""
        try:
            import os
            
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            checkpoint_path = os.path.join(models_dir, f"checkpoint_{checkpoint_type}_ep{episode}.pth")
            
            checkpoint = {
                'episode': episode,
                'global_step': self.global_step,
                'model_state_dict': self.agent.policy_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'avg_reward': self.best_avg_reward,
                'epsilon': self.agent.epsilon,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'episode_rewards': self.episode_rewards,
                'episode_losses': self.episode_losses,
                'episodes_without_improvement': self.episodes_without_improvement
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

    def _save_current_progress(self):
        """Save current progress khi interrupted"""
        try:
            self._save_checkpoint(self.episode, "interrupted")
            print("💾 Progress saved successfully")
        except Exception as e:
            print(f"❌ Error saving progress: {e}")

    def _force_refresh_data(self):
        """Force refresh data từ MongoDB"""
        try:
            self.mongo_manager._node_cache = {}
            self.mongo_manager.last_update_time = 0
            
            if hasattr(self.data_pipeline, '_current_snapshot'):
                self.data_pipeline._current_snapshot = None
            
            fresh_snapshot = self.mongo_manager.get_training_snapshot()
            
            if self.episode % 100 == 0:
                print(f"🔄 Data refreshed: {len(fresh_snapshot.get('nodes', {}))} nodes")
                
        except Exception as e:
            print(f"❌ Error refreshing data: {e}")

    def _log_process(self, episode: int, episode_reward: float, training_loss: float):
        """Enhanced logging"""
        buffer_size = self.buffer.get_size()
        
        # Calculate moving averages
        window_size = min(100, len(self.episode_rewards))
        if window_size > 0:
            recent_rewards = self.episode_rewards[-window_size:]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
        else:
            avg_reward = episode_reward
            reward_std = 0
        
        print(f"📊 Ep {episode:4d} | R: {episode_reward:7.2f} | Avg: {avg_reward:6.2f}±{reward_std:4.2f} | "
              f"L: {training_loss:6.4f} | ε: {self.agent.epsilon:.3f} | Buf: {buffer_size:5d}")
        
        # Periodic detailed logging
        if episode % 100 == 0:
            try:
                snapshot = self.data_pipeline.get_current_snapshot()
                nodes = snapshot.get('nodes', {})
                
                node_types = {}
                for node_data in nodes.values():
                    node_type = node_data.get('nodeType', 'UNKNOWN')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print(f"   🌐 Network: {', '.join([f'{k}:{v}' for k, v in node_types.items()])}")
                print(f"   📈 Progress: {episode/self.config.total_episodes*100:.1f}% | "
                      f"Best: {self.best_avg_reward:.2f} | No improve: {self.episodes_without_improvement}")
                
            except Exception as e:
                print(f"   ⚠️ Logging error: {e}")
