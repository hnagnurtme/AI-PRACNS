# agents/dpn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Optional
from env.action_mapper import ActionMapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 512):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class DqnAgent:
    """DQN Agent cho việc chọn node trong mạng SAGINs."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 5e-4, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.action_mapper: Optional[ActionMapper] = None
        self._learn_step_counter = 0
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        
        # Exploration startegy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    def select_action(self, state_vector: np.ndarray, available_nodes: List[str] = None, path_history: List[str] = None) -> str:
        """Chọn hành động với loop prevention"""
        if self.action_mapper is None:
            raise RuntimeError("ActionMapper chua duoc gan cho DqnAgent.")
        
        # Lấy danh sách nodes có thể chọn (tránh loops)
        valid_nodes = self._get_valid_nodes(available_nodes, path_history)
        
        if not valid_nodes:
            # Fallback: chọn random từ tất cả nodes (trường hợp khẩn cấp)
            return self.action_mapper.map_index_to_node_id(random.randrange(self.action_size))
        
        if random.random() < self.epsilon:
            # Exploration: chọn ngẫu nhiên từ valid nodes
            random_node = random.choice(valid_nodes)
            print(f"🎲 Exploration: chọn {random_node} từ {len(valid_nodes)} valid nodes")
            return random_node
        else:
            # Exploitation: Chọn hành động có Q-value cao nhất từ valid nodes
            state = torch.from_numpy(state_vector).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # Tìm action tốt nhất trong valid nodes
                best_value = -float('inf')
                best_action_index = None
                
                for i in range(self.action_size):
                    node_id = self.action_mapper.map_index_to_node_id(i)
                    if node_id in valid_nodes and q_values[0, i] > best_value:
                        best_value = q_values[0, i]
                        best_action_index = i
                
                if best_action_index is not None:
                    best_node = self.action_mapper.map_index_to_node_id(best_action_index)
                    print(f"🎯 Exploitation: chọn {best_node} (Q-value: {best_value:.3f})")
                    return best_node
                else:
                    # Fallback: chọn random từ valid nodes
                    fallback_node = random.choice(valid_nodes)
                    print(f"🔄 Fallback: chọn {fallback_node} (no valid Q-values)")
                    return fallback_node

    def _get_valid_nodes(self, available_nodes: List[str] = None, path_history: List[str] = None) -> List[str]:
        """Lấy danh sách nodes hợp lệ, tránh loops"""
        if available_nodes is None:
            # Nếu không có available_nodes, lấy tất cả nodes
            available_nodes = self.action_mapper.get_available_nodes()
        
        if path_history is None or not path_history:
            return available_nodes
        
        # Loại bỏ các nodes đã xuất hiện trong path history (tránh loops)
        valid_nodes = [node for node in available_nodes if node not in path_history]
        
        # Nếu không còn node nào hợp lệ, cho phép quay lại (nhưng sẽ bị penalty)
        if not valid_nodes:
            print("⚠️  No valid nodes left, allowing some repetition")
            return available_nodes[-3:]  # Chỉ cho phép 3 nodes gần nhất
        
        return valid_nodes

    def learn(self, experiences: List[Tuple[np.ndarray, str, float, np.ndarray]]) -> float:
        """Huấn luyện model từ một batch kinh nghiệm (off-policy)"""
        if self.action_mapper is None or len(experiences) < 2:  # FIX: cần ít nhất 2 samples cho batch norm
            return 0.0
        
        # Chuẩn bị dữ liệu 
        states_list, rewards_list, next_states_list, action_indices_list = [], [], [], []
        
        for s, a, r, sp in experiences:
            index = self.action_mapper.get_action_index(a)
            if index is not None:
                states_list.append(s)
                rewards_list.append(r)
                next_states_list.append(sp)
                action_indices_list.append(index)
        
        if len(states_list) < 2:  # FIX: cần ít nhất 2 samples
            return 0.0

        # Chuyển đổi sang tensor
        states = torch.as_tensor(np.array(states_list), dtype=torch.float32).to(DEVICE)
        rewards = torch.as_tensor(np.array(rewards_list), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.as_tensor(np.array(next_states_list), dtype=torch.float32).to(DEVICE)
        action_indices = torch.as_tensor(action_indices_list, dtype=torch.long).unsqueeze(1).to(DEVICE)
        
        # FIX: Set model to training mode
        self.policy_net.train()
        
        # Tính Q-values
        predicted_q_values = self.policy_net(states).gather(1, action_indices)
        
        # Tính target Q-values
        with torch.no_grad():
            self.target_net.eval()  # FIX: Set target net to eval mode
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values)
        
        # Tính loss và update 
        loss = self.criterion(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # FIX: Set back to eval mode for inference
        self.policy_net.eval()
        
        self._learn_step_counter += 1
        return loss.item()

    def soft_update_target_network(self, tau: float = 0.01):
        """Soft update target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self):
        """Cập nhật target network từ policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Giảm epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)