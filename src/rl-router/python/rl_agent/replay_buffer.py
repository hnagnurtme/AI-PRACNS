# python/rl_agent/replay_buffer.py

import numpy as np

class ReplayBuffer:
    """
    Bộ đệm tái trải nghiệm (Replay Buffer) được cấp phát bộ nhớ trước.
    
    Mục đích: Lưu trữ các "trải nghiệm" (state, action, reward, next_state, done)
    để agent có thể học từ một batch dữ liệu ngẫu nhiên, thay vì chỉ học
    từ trải nghiệm cuối cùng. Điều này giúp phá vỡ sự tương quan giữa
    các trải nghiệm liên tiếp và làm ổn định quá trình học.
    """

    def __init__(self, capacity: int, state_size: int):
        """
        Khởi tạo buffer.
        
        Args:
            capacity (int): Kích thước tối đa của buffer.
            state_size (int): Số chiều của vector trạng thái (state).
        """
        self.capacity = capacity
        self.state_size = state_size
        
        # --- Cấp phát trước bộ nhớ ---
        # Chúng ta tạo ra các mảng NumPy rỗng với kích thước tối đa (capacity).
        # Việc này hiệu quả hơn nhiều so với việc dùng list.append() của Python
        # vì không phải thay đổi kích thước mảng liên tục.
        
        # Mảng lưu trữ các trạng thái (S_t)
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        # Mảng lưu trữ các trạng thái kế tiếp (S_t+1)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        # Mảng lưu trữ các hành động (A_t)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        # Mảng lưu trữ các phần thưởng (R_t+1)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        # Mảng lưu trữ cờ báo kết thúc (True/False)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        # --- Con trỏ (Pointers) ---
        
        # 'position' là vị trí (index) mà dữ liệu MỚI sẽ được ghi vào.
        # Nó hoạt động như một con trỏ trong một "circular buffer" (bộ đệm vòng).
        self.position = 0
        
        # 'size' là số lượng trải nghiệm THỰC TẾ đang có trong buffer.
        # Nó sẽ tăng dần cho đến khi đạt giá trị 'capacity'.
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """
        Lưu một trải nghiệm mới vào buffer.
        Nếu buffer đầy, nó sẽ ghi đè lên trải nghiệm cũ nhất.
        """
        
        # Ghi dữ liệu mới vào vị trí 'position' hiện tại
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done) # Đảm bảo 'done' là float
        
        # Cập nhật vị trí 'position' cho lần ghi tiếp theo.
        # Phép toán modulo (%) giúp con trỏ tự động quay về 0
        # khi nó đạt đến 'capacity', tạo ra hiệu ứng "bộ đệm vòng".
        self.position = (self.position + 1) % self.capacity
        
        # Cập nhật kích thước thực tế của buffer.
        # self.size sẽ tăng cho đến khi nó bằng self.capacity,
        # sau đó nó sẽ giữ nguyên ở 'capacity'.
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Lấy ngẫu nhiên một batch (lô) các trải nghiệm từ buffer.
        """
        
        # Tạo ra một mảng các chỉ số (indices) ngẫu nhiên.
        # Chúng ta chỉ lấy mẫu từ 0 đến 'self.size' (số lượng mẫu đang có),
        # chứ không phải 'self.capacity' (kích thước tối đa).
        idx = np.random.choice(self.size, batch_size, replace=True)
        
        # (SỬA) Dùng `replace=True` (lấy mẫu có lặp lại):
        # Đây là cách làm an toàn và phổ biến. Nó cho phép chúng ta
        # lấy mẫu ngay cả khi 'batch_size' > 'self.size' (ví dụ: khi buffer
        # mới bắt đầu). Nếu dùng `replace=False`, code sẽ báo lỗi
        # trong trường hợp này.
        
        # Trả về các mảng numpy chứa batch dữ liệu tương ứng
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        """
        Hàm "magic method" này cho phép ta dùng hàm `len(buffer)`
        để lấy kích thước thực tế của buffer.
        """
        return self.size