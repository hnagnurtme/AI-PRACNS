# python/rl_agent/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# (NOTE) ĐỊNH NGHĨA HẰNG SỐ KIẾN TRÚC MỚI TẠI ĐÂY
INPUT_SIZE = 94    # (NOTE) TĂNG TỪ 52 LÊN 94
OUTPUT_SIZE = 10   # (NOTE) TĂNG TỪ 4 LÊN 10

class DQN(nn.Module):
    """Kiến trúc mạng DQN (MLP) với Dropout để tránh overfitting."""
    
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2):
        super(DQN, self).__init__()
        
        # (TỐI ƯU HÓA) Tăng kích thước hidden layers
        hidden_1 = 512   # Tăng từ 256
        hidden_2 = 256   # Tăng từ 128
        hidden_3 = 128   # Thêm lớp thứ 3
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_3, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass với Dropout, trả về Q-values cho 10 hành động."""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)


class DQNLegacy(nn.Module):
    """
    Legacy DQN architecture (3-layer) for loading old checkpoints.
    This matches the architecture used to train existing checkpoints.
    """
    
    def __init__(self, input_size: int, output_size: int, dropout_rate: float = 0.2):
        super(DQNLegacy, self).__init__()
        
        # Original architecture from training
        hidden_1 = 256
        hidden_2 = 128
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 3 layers, returns Q-values for 10 actions."""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def create_dqn_networks(dropout_rate: float = 0.2) -> tuple[DQN, DQN]:
    """
    Tạo Q-Network và Target-Network với cùng kiến trúc.
    Sử dụng hằng số INPUT_SIZE và OUTPUT_SIZE mới.
    
    :param dropout_rate: Tỷ lệ dropout (mặc định 0.2)
    :return: (q_network, target_network)
    """
    q_network = DQN(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network = DQN(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    
    # Đồng bộ hóa trọng số ban đầu
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval() # Target network chỉ dùng để inference
    
    return q_network, target_network


def create_legacy_dqn_networks(dropout_rate: float = 0.2) -> tuple[DQNLegacy, DQNLegacy]:
    """
    Create Q-Network and Target-Network with legacy 3-layer architecture.
    Use this to load old checkpoints trained with the original architecture.
    
    :param dropout_rate: Dropout rate (default 0.2)
    :return: (q_network, target_network)
    """
    q_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    
    # Sync initial weights
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()  # Target network for inference only
    
    return q_network, target_network