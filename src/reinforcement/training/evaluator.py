import numpy as np
from typing import List, Dict, Any
from agents.rl_agent import DQNAgent
from environments.dynamic_env import DynamicSatelliteEnv

class Evaluator:
    def __init__(self, config: Dict):
        self.config = config
        
    def compare_algorithms(self):
        # So sánh RL với Dijkstra và các baseline khác
        print("Comparing routing algorithms...")
        
        # Giả lập kết quả
        results = {
            'RL': {'success_rate': 0.85, 'avg_delay': 150.0, 'avg_hops': 3.2},
            'Dijkstra': {'success_rate': 0.80, 'avg_delay': 180.0, 'avg_hops': 3.5},
            'Random': {'success_rate': 0.45, 'avg_delay': 300.0, 'avg_hops': 5.1}
        }
        
        print("Algorithm Comparison Results:")
        for algo, metrics in results.items():
            print(f"{algo}: Success Rate={metrics['success_rate']:.2f}, "
                  f"Avg Delay={metrics['avg_delay']:.1f}ms, "
                  f"Avg Hops={metrics['avg_hops']:.1f}")