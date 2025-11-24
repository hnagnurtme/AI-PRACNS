import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .dqn_model import create_dqn_networks, create_legacy_dqn_networks, INPUT_SIZE, OUTPUT_SIZE
from .replay_buffer import ReplayBuffer
from .policy import get_epsilon, select_action 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, env, config: dict, use_legacy_architecture: bool = False):
        self.env = env
        self.config = config
        
        # Get hyperparameters from config
        self.gamma = self.config.get('gamma', 0.98)
        self.lr = self.config.get('learning_rate', 1e-4)
        self.batch_size = self.config.get('batch_size', 128)
        self.buffer_capacity = self.config.get('buffer_capacity', 200000)
        self.target_update_interval = self.config.get('target_update_interval', 50)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.05)
        self.epsilon_decay = self.config.get('epsilon_decay', 5000)
        
        if use_legacy_architecture:
            self.q_network, self.target_network = create_legacy_dqn_networks()
        else:
            self.q_network, self.target_network = create_dqn_networks()
        
        self.q_network.to(DEVICE)
        self.target_network.to(DEVICE)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = ReplayBuffer(self.buffer_capacity, INPUT_SIZE)

        self.steps_done = 0
        self.update_count = 0

    def select_action(self, state_vector: np.ndarray, is_training: bool = True, num_valid_actions: int = OUTPUT_SIZE) -> int:
        epsilon_params = {
            'start': self.epsilon_start,
            'end': self.epsilon_end,
            'decay': self.epsilon_decay
        }
        if not is_training:
            # Greedy action for evaluation
            return select_action(self.q_network, state_vector, 999999999, num_valid_actions, DEVICE, epsilon_params)
        else:
            # Epsilon-greedy action for training
            action = select_action(self.q_network, state_vector, self.steps_done, num_valid_actions, DEVICE, epsilon_params)
            self.steps_done += 1
            return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_batch = torch.from_numpy(states).float().to(DEVICE)
        next_state_batch = torch.from_numpy(next_states).float().to(DEVICE)
        action_batch = torch.from_numpy(actions).long().to(DEVICE)
        reward_batch = torch.from_numpy(rewards).float().to(DEVICE)
        done_batch = torch.from_numpy(dones).float().to(DEVICE)

        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1.0 - done_batch))

        loss = self.criterion(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint file."""
        self.q_network.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()  # Set to evaluation mode