# python/rl_agent/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# (NOTE) IMPORT KIẾN TRÚC MỚI
from python.rl_agent.dqn_model import create_dqn_networks, INPUT_SIZE, OUTPUT_SIZE
from python.rl_agent.replay_buffer import ReplayBuffer
from python.rl_agent.policy import get_epsilon  # (SỬA) Import hàm epsilon từ policy

# ======================== HYPERPARAMETERS ==========================
# (NOTE) Cập nhật các hằng số này
INPUT_SIZE = INPUT_SIZE      # 94
OUTPUT_SIZE = OUTPUT_SIZE    # 10

GAMMA = 0.95
LR = 5e-6
BATCH_SIZE = 64
BUFFER_CAPACITY = 100000
TARGET_UPDATE_INTERVAL = 10
EPSILON_DECAY_STEPS = 50000  # (SỬA) Giảm từ 10M xuống 50k để epsilon giảm nhanh hơn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== AGENT CLASS ==============================
class DQNAgent:
    def __init__(self, env):
        self.env = env
        
        # (NOTE) Tạo mạng 94-Input, 10-Output
        self.q_network, self.target_network = create_dqn_networks()
        self.q_network.to(device)
        self.target_network.to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        
        # (NOTE) Buffer giờ phải chứa state 94-chiều
        self.memory = ReplayBuffer(BUFFER_CAPACITY, INPUT_SIZE)

        self.steps_done = 0
        self.update_count = 0

    # ==============================================================
    def select_action(self, state_vector: np.ndarray) -> int:
        """Epsilon-Greedy Action Selection"""
        # (SỬA) Sử dụng hàm get_epsilon từ policy.py (thống nhất logic)
        epsilon = get_epsilon(self.steps_done)
        self.steps_done += 1

        if np.random.rand() < epsilon:
            # (NOTE) Khám phá ngẫu nhiên 1 trong 10 hành động
            return np.random.randint(OUTPUT_SIZE)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            
            # (NOTE) Lấy argmax từ 10 Q-values
            return q_values.argmax().item()

    # ==============================================================
    def optimize_model(self):
        """(TỐI ƯU) Cập nhật Q-network (đã sửa lỗi dtype)"""
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # (TỐI ƯU) Đảm bảo đúng dtype (float32)
        states = torch.tensor(states, device=device).float()
        next_states = torch.tensor(next_states, device=device).float()
        # (NOTE) actions giờ có thể từ 0-9
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).float().unsqueeze(1)
        dones = torch.tensor(dones, device=device).float().unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            # (NOTE) target_network(next_states) trả về 10 Q-values
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            q_targets = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.criterion(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % TARGET_UPDATE_INTERVAL == 0:
            self.update_target_network()

    # ==============================================================
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ==============================================================
    def save_checkpoint(self, path: str):
        torch.save(self.q_network.state_dict(), path)


# ======================== TRAINING LOOP ==========================
def train(agent: DQNAgent, env, num_episodes: int = 500):
    """(TỐI ƯU) Huấn luyện agent (đã loại bỏ logic thừa)"""
    for episode in range(num_episodes):
        packet_data = env.state_builder.generate_initial_packet()
        
        # (TỐI ƯU) env.simulate_episode đã tự quản lý việc
        # tối ưu và cập nhật target network bên trong
        total_reward = env.simulate_episode(agent, packet_data, max_hops=10)

        if episode % 10 == 0:
            print(f"[Episode {episode:03d}] Total Reward = {total_reward:.3f}")
        else:
             print(f"[Episode {episode:03d}] Total Reward = {total_reward:.3f}")