# python/rl_agent/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# (NOTE) ĐỊNH NGHĨA HẰNG SỐ KIẾN TRÚC MỚI TẠI ĐÂY
INPUT_SIZE = 94    # (NOTE) TĂNG TỪ 52 LÊN 94
OUTPUT_SIZE = 10   # (NOTE) TĂNG TỪ 4 LÊN 10

class DQN(nn.Module):
    """Kiến trúc mạng DQN (MLP)."""
    
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        
        # (NOTE) Kích thước các lớp có thể cần điều chỉnh
        # khi input/output thay đổi lớn
        hidden_1 = 256
        hidden_2 = 128
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, trả về Q-values cho 10 hành động."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def create_dqn_networks() -> tuple[DQN, DQN]:
    """
    Tạo Q-Network và Target-Network với cùng kiến trúc.
    Sử dụng hằng số INPUT_SIZE và OUTPUT_SIZE mới.
    """
    q_network = DQN(INPUT_SIZE, OUTPUT_SIZE)
    target_network = DQN(INPUT_SIZE, OUTPUT_SIZE)
    
    # Đồng bộ hóa trọng số ban đầu
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval() # Target network chỉ dùng để inference
    
    return q_network, target_network