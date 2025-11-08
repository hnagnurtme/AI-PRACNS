import torch
import random
import math
import numpy as np
from .dqn_model import DQN, OUTPUT_SIZE # Import OUTPUT_SIZE (10)

# --- HYPERPARAMETERS CHO EPSILON-GREEDY (TỐI ƯU HÓA) ---
EPS_START = 0.95      # Epsilon ban đầu (95% khám phá, tăng từ 90%)
EPS_END = 0.01        # Epsilon tối thiểu (1% khám phá, giảm từ 5% để tận dụng học được)
EPS_DECAY = 100000    # Tốc độ giảm epsilon (tăng từ 50000 để decay chậm hơn)

# --- ADAPTIVE EPSILON (MỚI) ---
def get_adaptive_epsilon(steps_done: int, performance_score: float = 0.0) -> float:
    """
    Tính epsilon với adaptive strategy dựa trên performance.
    
    :param steps_done: Số bước đã thực hiện
    :param performance_score: Điểm performance gần đây (0-1), càng cao càng tốt
    :return: Giá trị epsilon hiện tại
    """
    # Base epsilon theo exponential decay
    base_epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    # Nếu performance tốt (>0.7), giảm exploration thêm 20%
    if performance_score > 0.7:
        base_epsilon *= 0.8
    # Nếu performance kém (<0.3), tăng exploration thêm 20%
    elif performance_score < 0.3 and base_epsilon > EPS_END:
        base_epsilon = min(base_epsilon * 1.2, EPS_START)
    
    return base_epsilon


def get_epsilon(steps_done: int) -> float:
    """
    Tính toán giá trị epsilon hiện tại dựa trên số bước đã hoàn thành.
    Sử dụng hàm phân bố mũ giảm dần (standard approach).
    
    :param steps_done: Số bước đã hoàn thành
    :return: Giá trị epsilon (0 đến 1)
    """
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def select_action(q_network: DQN, state_vector: np.ndarray, steps_done: int, 
                 use_adaptive: bool = False, performance_score: float = 0.0) -> int:
    """
    Chọn hành động dựa trên chiến lược epsilon-greedy (với tùy chọn adaptive).
    
    :param q_network: Mạng Q chính (DQN).
    :param state_vector: Vector Trạng thái S hiện tại (NumPy array).
    :param steps_done: Tổng số bước đã hoàn thành.
    :param use_adaptive: Sử dụng adaptive epsilon hay không
    :param performance_score: Điểm performance để adaptive (nếu use_adaptive=True)
    :return: Index hành động được chọn (0-9).
    """
    if use_adaptive:
        epsilon = get_adaptive_epsilon(steps_done, performance_score)
    else:
        epsilon = get_epsilon(steps_done)
    
    if random.random() < epsilon:
        # Khám phá (Exploration): Chọn hành động ngẫu nhiên
        return random.randrange(OUTPUT_SIZE)
    else:
        # Khai thác (Exploitation): Chọn hành động có Q-Value lớn nhất
        with torch.no_grad():
            # Chuyển state NumPy sang Tensor và thêm chiều batch [1, 94]
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
            
            # Thực hiện forward pass và lấy index của giá trị lớn nhất (argmax)
            return q_network(state_tensor).argmax().item()