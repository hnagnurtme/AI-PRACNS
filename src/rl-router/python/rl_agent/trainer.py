# rl_agent/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from python.rl_agent.dqn_model import create_dqn_networks
from python.rl_agent.replay_buffer import ReplayBuffer

# ======================== HYPERPARAMETERS ==========================
INPUT_SIZE = 52
OUTPUT_SIZE = 4
GAMMA = 0.95
LR = 5e-5     # giảm nhẹ để học ổn định hơn
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_INTERVAL = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== AGENT CLASS ==============================
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.q_network, self.target_network = create_dqn_networks()
        self.q_network.to(device)
        self.target_network.to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(BUFFER_CAPACITY, INPUT_SIZE)

        self.steps_done = 0
        self.update_count = 0

    # ==============================================================
    def select_action(self, state_vector: np.ndarray) -> int:
        """Epsilon-Greedy Action Selection"""
        epsilon = 0.05 + 0.85 * np.exp(-1.0 * self.steps_done / 10000)
        self.steps_done += 1

        if np.random.rand() < epsilon:
            return np.random.randint(OUTPUT_SIZE)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    # ==============================================================
    def optimize_model(self):
        """Cập nhật Q-network"""
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.tensor(states, device=device)
        next_states = torch.tensor(next_states, device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, device=device).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
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
    """Huấn luyện agent với môi trường vệ tinh"""
    for episode in range(num_episodes):
        packet_data = env.state_builder.generate_initial_packet()  # hoặc mock 1 packet
        total_reward = env.simulate_episode(agent, packet_data, max_hops=10)

        if episode % 10 == 0:
            agent.update_target_network()

        print(f"[Episode {episode:03d}] Total Reward = {total_reward:.3f}")
