# agents/InMemoryReplayBuffer.py
import numpy as np
from typing import Dict, Any, List, Tuple
import random
import time

class InMemoryReplayBuffer:
    """Buffer lưu trữ kinh nghiệm trong bộ nhớ."""
    
    def __init__(self, capacity: int = 100000):
        self.memory: List[Tuple[np.ndarray, str, float, np.ndarray]] = []
        self.capacity = capacity
        self.position = 0

    def store_experience(self, experience: Dict):
        """Lưu một kinh nghiệm mới (S, A, R, S')."""
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

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, str, float, np.ndarray]]:
        """Lấy ngẫu nhiên một batch kinh nghiệm từ bộ nhớ."""
        if len(self.memory) < batch_size:
            return []
            
        return random.sample(self.memory, batch_size)

    def get_size(self) -> int:
        """Trả về số lượng kinh nghiệm hiện tại."""
        return len(self.memory)