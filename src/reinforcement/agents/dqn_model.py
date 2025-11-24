import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 162   
OUTPUT_SIZE = 10   

class DuelingDQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DuelingDQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DQNLegacy(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2):
        super(DQNLegacy, self).__init__()
        hidden_1 = 256
        hidden_2 = 128
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def create_dqn_networks(dropout_rate: float = 0.0) -> tuple:
    q_network = DuelingDQN(INPUT_SIZE, OUTPUT_SIZE)
    target_network = DuelingDQN(INPUT_SIZE, OUTPUT_SIZE)
    
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    return q_network, target_network

def create_legacy_dqn_networks(dropout_rate: float = 0.2) -> tuple:
    q_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    return q_network, target_network