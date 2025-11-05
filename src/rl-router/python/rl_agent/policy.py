import torch
import random
import math
import numpy as np
from .dqn_model import DQN, OUTPUT_SIZE # Import OUTPUT_SIZE (4)

# --- HYPERPARAMETERS CHO EPSILON-GREEDY ---
EPS_START = 0.9      # Epsilon ban đầu (90% khám phá)
EPS_END = 0.05       # Epsilon tối thiểu (5% khám phá)
EPS_DECAY = 50000    # (SỬA) Tốc độ giảm epsilon (đồng bộ với trainer.py)

def get_epsilon(steps_done: int) -> float:
    """
    Tính toán giá trị epsilon hiện tại dựa trên số bước đã hoàn thành.
    Sử dụng hàm phân bố mũ giảm dần.
    """
    # Epsilon = EPS_END + (EPS_START - EPS_END) * exp(-1 * steps_done / EPS_DECAY)
    
    return EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)


def select_action(q_network: DQN, state_vector: np.ndarray, steps_done: int) -> int:
    """
    Chọn hành động dựa trên chiến lược epsilon-greedy.
    
    :param q_network: Mạng Q chính (DQN).
    :param state_vector: Vector Trạng thái S hiện tại (NumPy array).
    :param steps_done: Tổng số bước đã hoàn thành.
    :return: Index hành động được chọn (0, 1, 2, hoặc 3).
    """
    epsilon = get_epsilon(steps_done)
    
    if random.random() < epsilon:
        # Khám phá (Exploration): Chọn hành động ngẫu nhiên
        return random.randrange(OUTPUT_SIZE)
    else:
        # Khai thác (Exploitation): Chọn hành động có Q-Value lớn nhất
        with torch.no_grad():
            # Chuyển state NumPy sang Tensor và thêm chiều batch [1, 52]
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
            
            # Thực hiện forward pass và lấy index của giá trị lớn nhất (argmax)
            return q_network(state_tensor).argmax().item()