# agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, output_size)  # Fixed typo: advange_stream -> advantage_stream
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class ActionMapper:
    def __init__(self):
        self.node_to_index = {}
        self.index_to_node = {}
        
    def update_nodes(self, all_nodes: List[str]):
        self.node_to_index = {
            node: i for i, node in enumerate(sorted(all_nodes))
        }
        self.index_to_node = {
            i: node for node, i in self.node_to_index.items()
        }
    
    def get_action_index(self, node_id: str) -> Optional[int]:
        return self.node_to_index.get(node_id)

    def map_index_to_node_id(self, index: int) -> Optional[str]:
        return self.index_to_node.get(index)  # Fixed typo: index_to_node.get(index)

    def get_action_size(self) -> int:
        return len(self.node_to_index)
    
class DqnAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-4, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.policy_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_mapper = ActionMapper()
        self.tau = 0.001
        self.checkpoint_path = "rl_checkpoint.pth"
    
    def select_action(self, state: np.ndarray, valid_actions: List[str]) -> str:  # Fixed: List[int] -> List[str], return str
        try:
            self.action_mapper.update_nodes(valid_actions)
            if random.random() < self.epsilon:
                return random.choice(valid_actions)
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor).squeeze(0)
                valid_indices = [self.action_mapper.get_action_index(a) for a in valid_actions if self.action_mapper.get_action_index(a) is not None]
                if not valid_indices:
                    logger.warning("No valid indices, fallback to random")
                    return random.choice(valid_actions)
                valid_q = q_values[valid_indices]
                best_index = valid_indices[valid_q.argmax().item()]
                return self.action_mapper.map_index_to_node_id(best_index)
        except Exception as e:
            logger.error(f"Error updating action mapper: {e}")
            return random.choice(valid_actions)
        
    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray]]):
        try:
            if len(experiences) == 0:
                return 0.0
            states, actions, rewards, next_states = zip(*experiences)
            states = torch.from_numpy(np.array(states)).float().to(DEVICE)
            actions = torch.tensor(actions).unsqueeze(1).to(DEVICE)
            rewards = torch.tensor(rewards).unsqueeze(1).to(DEVICE)
            next_states = torch.from_numpy(np.array(next_states)).float().to(DEVICE)
            
            # Double DQN
            with torch.no_grad():
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q = self.target_net(next_states).gather(1, next_actions)
            
            target_q = rewards + self.gamma * next_q
            q_values = self.policy_net(states).gather(1, actions)
            
            loss = self.criterion(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except Exception as e:
            logger.error(f"Error in learning step: {e}")
            return 0.0
    
    def soft_update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self):
        checkpoint = {
            'state_dict': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved at {self.checkpoint_path}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['state_dict'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
        else:
            logger.info("No checkpoint found, starting fresh")