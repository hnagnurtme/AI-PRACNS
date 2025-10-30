# python/rl_agent/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from python.rl_agent.dqn_model import create_dqn_networks, DQN
from python.rl_agent.replay_buffer import ReplayBuffer
import numpy as np

INPUT_SIZE = 52
OUTPUT_SIZE = 4
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def select_action(self, state_vector: np.ndarray):
        epsilon = 0.05 + 0.85 * np.exp(-1.0 * self.steps_done / 10000)
        self.steps_done += 1
        if np.random.rand() < epsilon:
            return np.random.randint(OUTPUT_SIZE)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.tensor(states, device=device)
        next_states = torch.tensor(next_states, device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, device=device).unsqueeze(1)

        q_expected = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.criterion(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path: str):
        torch.save(self.q_network.state_dict(), path)
