from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseEnv(ABC):
    @abstractmethod
    def reset(self, initial_packet_data: Dict[str, Any]) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple:
        pass

    @abstractmethod
    def simulate_episode(self, agent, initial_packet_data: Dict[str, Any], max_hops: int) -> Dict[str, Any]:
        pass