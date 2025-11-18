import torch
import torch.nn as nn
import torch.nn.functional as F

# --- HẰNG SỐ KIẾN TRÚC ---
# Input 162: Do padding thêm 2 số 0 ở main_train.py để khớp code cũ
# Input chuẩn nên là 160 (Self 12 + Dest 8 + Neighbor 140)
INPUT_SIZE = 162   
OUTPUT_SIZE = 10   

class DuelingDQN(nn.Module):
    """
    Kiến trúc Dueling DQN (Cải tiến): Tách biệt Value và Advantage.
    Loại bỏ Dropout để tăng tính ổn định khi hội tụ Q-Learning.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super(DuelingDQN, self).__init__()
        
        # --- Feature Extractor (Phần chung) ---
        # Trích xuất đặc trưng từ trạng thái thô
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # --- Stream 1: Value Function V(s) ---
        # Đánh giá giá trị của bản thân trạng thái này (Good/Bad state)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output là 1 số vô hướng (Scalar)
        )
        
        # --- Stream 2: Advantage Function A(s, a) ---
        # Đánh giá lợi thế của từng hành động so với trung bình
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size) # Output là vector Advantage cho 10 actions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Feature Extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 2. Split Streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # 3. Aggregation (Kết hợp)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Trừ đi mean để đảm bảo tính định danh (Identifiability)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DQNLegacy(nn.Module):
    """
    [Legacy] Kiến trúc cũ dùng để load checkpoint đã train trước đó.
    Giữ nguyên Dropout để khớp keys trong state_dict.
    """
    
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

def create_dqn_networks(dropout_rate: float = 0.0) -> tuple[DuelingDQN, DuelingDQN]:
    """
    Tạo Dueling Q-Network (Policy & Target).
    Lưu ý: dropout_rate không còn được dùng trong DuelingDQN mới.
    """
    q_network = DuelingDQN(INPUT_SIZE, OUTPUT_SIZE)
    target_network = DuelingDQN(INPUT_SIZE, OUTPUT_SIZE)
    
    # Đồng bộ hóa trọng số ban đầu
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval() # Target network luôn ở chế độ eval
    
    return q_network, target_network

def create_legacy_dqn_networks(dropout_rate: float = 0.2) -> tuple[DQNLegacy, DQNLegacy]:
    """Hàm factory cho kiến trúc cũ"""
    q_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network = DQNLegacy(INPUT_SIZE, OUTPUT_SIZE, dropout_rate)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    return q_network, target_network