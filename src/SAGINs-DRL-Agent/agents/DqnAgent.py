# agents/DqnAgent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Optional
from env.ActionMapper import ActionMapper 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DqnAgent:
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 5e-4, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.action_mapper: Optional[ActionMapper] = None 
        self._learn_step_counter = 0

        self.policy_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

    def select_action(self, state_vector: np.ndarray) -> str:
        """Chọn hành động bằng Epsilon-Greedy và trả về Node ID."""
        if self.action_mapper is None:
            raise RuntimeError("ActionMapper chưa được gán cho DqnAgent.")

        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state_vector).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_index = q_values.max(1)[1].item()
        
        return self.action_mapper.map_index_to_node_id(action_index)

    def learn(self, experiences: List[Tuple[np.ndarray, str, float, np.ndarray]]):
        """Huấn luyện mô hình từ một batch kinh nghiệm (Off-policy)."""
        
        if self.action_mapper is None: return 0.0

        states_list, rewards_list, next_states_list, action_indices_list = [], [], [], []
        
        for s, a, r, sp in experiences:
            index = self.action_mapper.get_action_index(a)
            if index is not None:
                states_list.append(s); rewards_list.append(r); next_states_list.append(sp); action_indices_list.append(index)

        if not states_list: return 0.0

        # Tạo Tensor và Chuyển sang DEVICE
        states = torch.as_tensor(np.array(states_list), dtype=torch.float32).to(DEVICE)
        rewards = torch.as_tensor(np.array(rewards_list), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.as_tensor(np.array(next_states_list), dtype=torch.float32).to(DEVICE)
        action_indices = torch.tensor(action_indices_list, dtype=torch.long).unsqueeze(1).to(DEVICE)

        # Predicted Q-value: Q(s, a)
        predicted_q_values = self.policy_net(states).gather(1, action_indices)

        # Target Q-value: R + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values)

        # Tính Loss và Tối ưu hóa
        loss = self.criterion(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        self._learn_step_counter += 1
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)