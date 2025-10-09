# simulator/NetworkSimulator.py
from agents.DqnAgent import DqnAgent
from agents.InMemoryReplayBuffer import InMemoryReplayBuffer
from env.StateProcessor import StateProcessor
from env.RewardCalculator import RewardCalculator
from typing import Dict, Any
import time
import numpy as np 
from utils.static_data import BASE_NODE_INFO, STATIC_LINK_VARIANTS 
import random 

class NetworkSimulator:
    
    def __init__(self, agent: DqnAgent, buffer: InMemoryReplayBuffer):
        self.agent = agent
        self.buffer = buffer
        self.processor = StateProcessor()
        self.reward_calc = RewardCalculator()
        
        self.default_qos: Dict[str, Any] = {
            "serviceType": "VIDEO_STREAMING",
            "maxLatencyMs": 50.0,
            "minBandwidthMbps": 500.0,
            "maxLossRate": 0.02
        }
        
    def _collect_current_state_data(self, source_id: str, dest_id: str) -> Dict[str, Any]:
        """Tải dữ liệu TĨNH từ bộ nhớ (chọn NGẪU NHIÊN một biến thể LinkMetrics)."""
        
        src_info = BASE_NODE_INFO.get(source_id)
        
        random_variant = random.choice(STATIC_LINK_VARIANTS)
        neighbor_links_list = [link for link in random_variant if link['sourceNodeId'] == source_id]
        neighbor_link_metrics = {link['destinationNodeId']: link for link in neighbor_links_list}
        
        return {
            "sourceNodeId": source_id,
            "destinationNodeId": dest_id,
            "targetQoS": self.default_qos,
            "sourceNodeInfo": src_info if src_info else {},
            "neighborLinkMetrics": neighbor_link_metrics 
        }
    
    def simulate_one_step(self, source_id: str, dest_id: str) -> Dict[str, Any]:
        """Thực hiện một bước tương tác Agent-Môi trường."""
        
        raw_state_s = self._collect_current_state_data(source_id, dest_id) 
        state_vector_s = self.processor.json_to_state_vector(raw_state_s)
        
        action_id_a = self.agent.select_action(state_vector_s)
        
        chosen_link_metric = raw_state_s['neighborLinkMetrics'].get(action_id_a)
        reward_r = self.reward_calc.calculate_reward(self.default_qos, chosen_link_metric if chosen_link_metric else {})
        
        # State S' (Lấy một biến thể trạng thái mới)
        raw_state_s_prime = self._collect_current_state_data(source_id, dest_id) 
        state_vector_s_prime = self.processor.json_to_state_vector(raw_state_s_prime)
        
        # Lưu Experience
        experience = {
            "timestamp": int(time.time() * 1000),
            "sourceNodeId": source_id,
            "serviceType": self.default_qos['serviceType'],
            "stateVectorS": state_vector_s.tolist(),
            "actionTakenA": action_id_a,
            "rewardR": reward_r,
            "nextStateVectorSPrime": state_vector_s_prime.tolist()
        }
        self.buffer.store_experience(experience)
        
        return {"action_taken": action_id_a, "reward": reward_r}

    def run_exploration_loop(self, num_steps: int, source: str, dest: str):
        """Chạy vòng lặp Khám phá để lấp đầy Buffer."""
        print(f"\n--- Bắt đầu giai đoạn KHÁM PHÁ (Epsilon: {self.agent.epsilon:.3f}) ---")
        for i in range(num_steps):
            result = self.simulate_one_step(source_id=source, dest_id=dest) 
            if (i + 1) % 100 == 0:
                print(f"Bước {i+1}/{num_steps}: Hành động: {result['action_taken']}, R: {result['reward']:.2f}, Buffer Size: {self.buffer.get_size()}")