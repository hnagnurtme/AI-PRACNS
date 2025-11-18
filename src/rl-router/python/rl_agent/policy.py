import torch
import torch.nn as nn
import random
import math
import numpy as np

# Import hằng số kích thước output
from .dqn_model import OUTPUT_SIZE

# --- HYPERPARAMETERS ---
EPS_START = 1.0       # Bắt đầu khám phá 100% (để Agent đi lung tung gom data)
EPS_END = 0.05        # Giữ lại 5% ngẫu nhiên để không bị kẹt vào cục bộ
EPS_DECAY = 5000      # Giảm nhanh hơn (100k là quá chậm nếu train 1000 episode)

def get_epsilon(steps_done: int) -> float:
    """
    Tính Epsilon theo hàm mũ giảm dần.
    """
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

def select_action(q_network: nn.Module, 
                  state_vector: np.ndarray, 
                  steps_done: int, 
                  num_valid_actions: int, 
                  device: torch.device) -> int:
    """
    Chọn hành động có Action Masking & Device Handling.
    
    :param q_network: Mạng DuelingDQN đang train.
    :param state_vector: Numpy array input (162,).
    :param steps_done: Số bước train hiện tại.
    :param num_valid_actions: Số lượng neighbor thực tế của node hiện tại.
    :param device: 'cpu' hoặc 'cuda'.
    """
    epsilon = get_epsilon(steps_done)
    
    # --- 1. EXPLORATION (Khám phá ngẫu nhiên) ---
    if random.random() < epsilon:
        # QUAN TRỌNG: Chỉ random trong số neighbor có thật
        # Nếu num_valid_actions = 3, chỉ chọn 0, 1, 2
        if num_valid_actions <= 0: return 0 # Fallback an toàn
        return random.randint(0, num_valid_actions - 1)

    # --- 2. EXPLOITATION (Khai thác Q-Value tốt nhất) ---
    else:
        with torch.no_grad():
            # Chuyển sang Tensor và đưa vào đúng Device (GPU/CPU)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            
            # Forward pass lấy Q-values [Batch=1, Actions=10]
            q_values = q_network(state_tensor)
            
            # --- ACTION MASKING ---
            # Kỹ thuật này đảm bảo Agent không bao giờ chọn neighbor ảo
            # ngay cả khi mạng Neural dự đoán sai giá trị cao cho nó.
            
            # Tạo mask với giá trị -Infinity
            mask = torch.full_like(q_values, float('-inf'))
            
            # Mở mask cho các action hợp lệ (gán về 0)
            # q_values + 0 = q_values (giữ nguyên)
            # q_values + (-inf) = -inf (bị loại bỏ)
            safe_actions = min(num_valid_actions, OUTPUT_SIZE)
            mask[:, :safe_actions] = 0
            
            masked_q_values = q_values + mask
            
            # Chọn index có giá trị lớn nhất
            return masked_q_values.argmax(dim=1).item()