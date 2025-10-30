# python/rl_agent/dqn_model.py

import torch
import torch.nn as nn

INPUT_SIZE = 52
OUTPUT_SIZE = 4

class DQN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output_layer(x)

def create_dqn_networks():
    q_network = DQN(INPUT_SIZE, OUTPUT_SIZE)
    target_network = DQN(INPUT_SIZE, OUTPUT_SIZE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    return q_network, target_network
