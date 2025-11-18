import numpy as np

class ReplayBuffer:
    """
    Bộ đệm tái trải nghiệm (Replay Buffer) tối ưu hóa cho DQN.
    Lưu trữ dữ liệu trên RAM (NumPy) để tiết kiệm VRAM GPU.
    """

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        
        # Pre-allocate memory (Giữ nguyên vì đã tối ưu)
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        
        # Action là số nguyên (Index của neighbor)
        self.actions = np.zeros((capacity, 1), dtype=np.int64) 
        # Reward và Done lưu dạng cột (N, 1) ngay từ đầu để tránh lỗi broadcasting
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Lưu experience vào buffer."""
        # Đảm bảo state đúng shape
        state = np.array(state)
        next_state = np.array(next_state)

        self.states[self.position] = state
        self.next_states[self.position] = next_state
        
        # Lưu các giá trị scalar vào mảng 2D (N, 1)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        
        # Cập nhật con trỏ vòng
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Lấy mẫu ngẫu nhiên.
        Trả về các mảng NumPy đã được định hình đúng chuẩn (Batch, Dim).
        """
        # Dùng randint nhanh hơn choice
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[idxs],      # Shape: (Batch, State_Size)
            self.actions[idxs],     # Shape: (Batch, 1)
            self.rewards[idxs],     # Shape: (Batch, 1)
            self.next_states[idxs], # Shape: (Batch, State_Size)
            self.dones[idxs]        # Shape: (Batch, 1)
        )

    def __len__(self):
        return self.size