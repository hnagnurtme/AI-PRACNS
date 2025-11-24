import numpy as np
from typing import List, Optional
from .rl_agent import DQNAgent


class MultiAgent:
    def __init__(self, num_agents: int, env, use_legacy_architecture: bool = False):
        self.agents: List[DQNAgent] = []
        for i in range(num_agents):
            self.agents.append(DQNAgent(env, {"use_legacy_architecture": use_legacy_architecture}))
    
    def select_actions(self, state_vectors: List[np.ndarray], greedy: bool = False, num_valid_actions_list: Optional[List[int]] = None) -> List[int]:
        actions = []
        for i, state in enumerate(state_vectors):
            num_actions = num_valid_actions_list[i] if num_valid_actions_list else 10
            actions.append(self.agents[i].select_action(state, greedy, num_actions))
        return actions
    
    def optimize_models(self):
        for agent in self.agents:
            agent.optimize_model()