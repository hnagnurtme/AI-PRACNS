# agents/replay_buffer.py
from collections import deque
import random
from typing import List, Tuple
import numpy as np

class ReplayBuffer:
    def __init__ (self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray]):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        if (len(self.buffer) < batch_size):
            return []
        return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        return len(self.buffer)