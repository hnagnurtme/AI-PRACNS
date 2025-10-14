# simulator/network_simulator.py
import time
from typing import Dict, Any, List
import numpy as np
import random
from agents.dpn_agent import DqnAgent
from agents.replay_buffer import InMemoryReplayBuffer as ReplayBuffer
from data.mongo_manager import MongoDataManager
from data.data_pipeline import RealTimeDataPipeline as DataPipeline
from env.link_metrics_calculator import LinkMetricsCalculator
from env.reward_calculator import RewardCalculator
from env.state_processor import StateProcessor

class NetworkSimulator:
    """Simulator tương tác với môi trường SAGINs"""

    def __init__(self, agent: DqnAgent, buffer: ReplayBuffer, data_pipeline: DataPipeline, link_calculator: LinkMetricsCalculator, reward_calculator: RewardCalculator, state_processor: StateProcessor):
        self.agent = agent
        self.buffer = buffer
        self.data_pipeline = data_pipeline
        self.link_calculator = link_calculator
        self.reward_calculator = reward_calculator
        self.state_processor = state_processor
        
        self.default_qos = {
            "serviceType": "VIDEO_STREAMING",
            'maxLatencyMs': 50.0,
            'minBandwidthMbps': 500.0,
            'maxLossRate': 0.02
        }
        
    def simulation_one_step(self, source_id: str, dest_id: str, path_history: List[str] = None, current_step: int = 0, max_steps: int = 10):
        """Thực hiện 1 bước trong mô phỏng với improved action selection"""
        try:
            # 1. Thu thập state hiện tại 
            raw_state_s = self._collect_current_state(source_id, dest_id)
            
            # 2. Chuyển thành state vector
            state_vector_s = self.state_processor.json_to_state_vector(raw_state_s)
            
            # 3. Lấy danh sách neighbors có thể chọn
            neighbors_links = raw_state_s.get('neighborLinkMetrics', {})
            available_nodes = list(neighbors_links.keys())
            
            # 4. Agent chọn next hop với loop prevention
            next_hop_id = self.agent.select_action(
                state_vector_s, 
                available_nodes=available_nodes,
                path_history=path_history
            )
            
            print(f"🔄 Step {current_step}/{max_steps}: {source_id} -> {next_hop_id} (path: {path_history})")
            
            # 5. Lấy thông tin next hop node
            snapshot = self.data_pipeline.mongo_manager.get_training_snapshot()
            next_hop_node = snapshot.get('nodes', {}).get(next_hop_id, {})
            
            # 6. Tính links metrics
            source_node = raw_state_s.get('sourceNodeInfo', {})
            chosen_link_metrics = self.link_calculator.calculate_link_metrics(source_node, next_hop_node)
            
            # 7. Tính reward
            reached_destination = (next_hop_id == dest_id)
            reward_r = self.reward_calculator.calculate_reward(
                self.default_qos,
                chosen_link_metrics,
                source_node,
                next_hop_node,
                step=current_step,
                total_steps=max_steps,
                reached_destination=reached_destination,
                path_history=path_history
            )
            
            # 8. Thu thập state tiếp theo
            raw_state_s_prime = self._collect_current_state(next_hop_id, dest_id)
            state_vector_s_prime = self.state_processor.json_to_state_vector(raw_state_s_prime)
            
            # 9. Lưu kinh nghiệm vào Replay Buffer
            experience = {
                "timestamp": int(time.time() * 1000),
                "sourceNodeId": source_id,
                "destinationNodeId": dest_id,
                "serviceType": self.default_qos['serviceType'],
                "stateVectorS": state_vector_s.tolist(),
                "actionTakenA": next_hop_id,
                "rewardR": reward_r,
                "nextStateVectorSPrime": state_vector_s_prime.tolist()
            }
            self.buffer.store_experience(experience)
        
            return {
                "action_taken": next_hop_id,
                "reward": reward_r,
                "source": source_id,
                "destination": dest_id,
                "link_metrics": chosen_link_metrics,
                "next_state": raw_state_s_prime,
                "reached_destination": reached_destination
            }
        except Exception as e:
            print(f"❌ Error in simulation step: {e}")
            import traceback
            traceback.print_exc()
            return {
                "action_taken": dest_id,
                "reward": -20.0,
                'reached_destination': False
            }
    
    def _collect_current_state(self, source_id: str, dest_id: str) -> Dict[str, Any]:
        """Thu thập state hiện tại từ node info"""
        snapshot = self.data_pipeline.get_current_snapshot()
        nodes = snapshot['nodes']
        
        source_node = nodes.get(source_id)
        dest_node = nodes.get(dest_id)
        
        if not source_node or not dest_node:
            raise ValueError(f"Node not found: {source_id} or {dest_id}")
        
        # Tính link metrics real-time cho tất cả neighbors
        neighbor_links = {}
        for neighbor_id, neighbor_node in nodes.items():
            if neighbor_id != source_id:
                link_metrics = self.link_calculator.calculate_link_metrics(source_node, neighbor_node)
                if link_metrics['isLinkActive']:
                    neighbor_links[neighbor_id] = link_metrics
        
        return {
            "sourceNodeId": source_id,
            "destinationNodeId": dest_id,
            "targetQoS": self.default_qos,
            "sourceNodeInfo": source_node,
            "destinationNodeInfo": dest_node,
            "neighborLinkMetrics": neighbor_links
        }

    def run_exploration_phase(self, num_steps: int, source_nodes: List[str], dest_nodes: List[str]):
        """Chạy phase khám phá"""
        print(f"🔍 Starting exploration: {num_steps} steps")
        
        for step in range(num_steps):
            source = random.choice(source_nodes)
            dest = random.choice(dest_nodes)
            
            result = self.simulate_one_step(source, dest)
            
            if (step + 1) % 100 == 0:
                print(f"Exploration {step+1}/{num_steps}: {source}→{result['action_taken']} "
                    f"Reward: {result['reward']:.2f}")

            