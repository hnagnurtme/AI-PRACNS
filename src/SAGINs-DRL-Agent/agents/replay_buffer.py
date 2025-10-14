# agents/replay_buffer.py
import numpy as np
import random
from typing import Dict, Any, List, Tuple

class InMemoryReplayBuffer:
    """Bộ đệm replay buffer trong bộ nhớ"""
    
    def __init__(self, capacity: int = 100000):
        # XÓA dòng này: super().__init__(capacity)
        self.capacity = capacity
        self.memory: List[Tuple[np.ndarray, str, float, np.ndarray]] = []
        self.position = 0
        print(f"🔧 ReplayBuffer initialized: capacity={capacity}")
    
    def store_experience(self, experience: Dict):
        """Lưu experience mới"""
        try:
            exp_tuple = (
                np.array(experience['stateVectorS'], dtype=np.float32),
                experience['actionTakenA'],
                experience['rewardR'],
                np.array(experience['nextStateVectorSPrime'], dtype=np.float32)
            )
            
            if len(self.memory) < self.capacity:
                self.memory.append(exp_tuple)
            else:
                self.memory[self.position] = exp_tuple
            self.position = (self.position + 1) % self.capacity
            
            # Debug log mỗi 1000 experiences
            if len(self.memory) % 1000 == 0:
                print(f"💾 Buffer size: {len(self.memory)}/{self.capacity}")
                
        except Exception as e:
            print(f"❌ Error storing experience: {e}")

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, str, float, np.ndarray]]:
        """Lấy batch experiences ngẫu nhiên"""
        if len(self.memory) < batch_size:
            if len(self.memory) > 0:
                # Nếu buffer chưa đủ batch_size, trả về tất cả
                return self.memory.copy()
            return []
        
        try:
            samples = random.sample(self.memory, batch_size)
            return samples
        except Exception as e:
            print(f"❌ Error sampling from buffer: {e}")
            return []

    def get_size(self) -> int:
        """Trả về số lượng experiences hiện có"""
        return len(self.memory)

    def get_capacity(self) -> int:
        """Trả về dung lượng tối đa"""
        return self.capacity

    def clear(self):
        """Xóa toàn bộ buffer"""
        self.memory.clear()
        self.position = 0
        print("🧹 Replay buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về buffer"""
        if not self.memory:
            return {"size": 0, "rewards": []}
        
        rewards = [exp[2] for exp in self.memory]  # reward là phần tử thứ 2
        return {
            "size": len(self.memory),
            "reward_mean": np.mean(rewards) if rewards else 0,
            "reward_std": np.std(rewards) if rewards else 0,
            "reward_min": min(rewards) if rewards else 0,
            "reward_max": max(rewards) if rewards else 0
        }