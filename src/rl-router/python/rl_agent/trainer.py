import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import modules
from python.rl_agent.dqn_model import (
    create_dqn_networks, 
    create_legacy_dqn_networks,
    INPUT_SIZE, 
    OUTPUT_SIZE
)
from python.rl_agent.replay_buffer import ReplayBuffer
# Import cả select_action để dùng logic masking chuẩn
from python.rl_agent.policy import get_epsilon, select_action 

# ======================== HYPERPARAMETERS ==========================
# Input size nên là 162 (theo dqn_model hiện tại)
GAMMA = 0.98                 # Tăng nhẹ để nhìn xa hơn (Long-term routing)
LR = 1e-4                    # Tăng Learning Rate (3e-6 quá chậm cho Dueling DQN)
BATCH_SIZE = 128
BUFFER_CAPACITY = 200000
TARGET_UPDATE_INTERVAL = 50  # Cập nhật target chậm lại chút để ổn định
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, env, use_legacy_architecture: bool = False):
        self.env = env
        
        # 1. Khởi tạo Mạng Neural
        if use_legacy_architecture:
            self.q_network, self.target_network = create_legacy_dqn_networks()
        else:
            # Dueling DQN (Không cần dropout rate nữa)
            self.q_network, self.target_network = create_dqn_networks()
        
        self.q_network.to(DEVICE)
        self.target_network.to(DEVICE)

        # 2. Optimizer & Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        # Dùng SmoothL1Loss (Huber Loss) tốt hơn MSE cho RL
        self.criterion = nn.SmoothL1Loss()
        
        # 3. Replay Buffer
        self.memory = ReplayBuffer(BUFFER_CAPACITY, INPUT_SIZE)

        self.steps_done = 0
        self.update_count = 0

    def select_action(self, state_vector: np.ndarray, greedy: bool = False, num_valid_actions: int = OUTPUT_SIZE) -> int:
        """
        Wrapper gọi hàm select_action từ policy.py.
        Giúp logic thống nhất và tận dụng Action Masking chuẩn.
        """
        if greedy:
            # Nếu greedy (dùng khi test), ép epsilon về 0
            # Ta gọi trực tiếp logic exploitation của policy hoặc giả lập bước rất lớn
            return select_action(self.q_network, state_vector, 999999999, num_valid_actions, DEVICE)
        else:
            # Training bình thường
            action = select_action(self.q_network, state_vector, self.steps_done, num_valid_actions, DEVICE)
            self.steps_done += 1
            return action

    def optimize_model(self):
        """Cập nhật trọng số mạng DQN."""
        if len(self.memory) < BATCH_SIZE:
            return

        # 1. Sample từ Buffer (Lưu ý: Buffer đã trả về đúng shape (N,1))
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 2. Chuyển sang Tensor & Đưa lên GPU
        # Dùng .from_numpy() nhanh hơn .tensor() vì nó chia sẻ bộ nhớ nếu có thể
        state_batch = torch.from_numpy(states).float().to(DEVICE)
        next_state_batch = torch.from_numpy(next_states).float().to(DEVICE)
        action_batch = torch.from_numpy(actions).long().to(DEVICE) # Actions phải là Long/Int64
        reward_batch = torch.from_numpy(rewards).float().to(DEVICE)
        done_batch = torch.from_numpy(dones).float().to(DEVICE)

        # 3. Tính Q(s, a) hiện tại
        # action_batch shape (128, 1) -> gather ok
        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        # 4. Tính V(s') từ Target Network (Double DQN logic could be added here)
        with torch.no_grad():
            # Lấy max Q-value của trạng thái tiếp theo
            # .max(1)[0] trả về (128,), cần unsqueeze(1) thành (128, 1) để cộng với reward
            next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            
            # Bellman Equation: Target = Reward + Gamma * MaxQ * (1 - Done)
            expected_q_values = reward_batch + (GAMMA * next_q_values * (1.0 - done_batch))

        # 5. Tính Loss & Backprop
        loss = self.criterion(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Chống bùng nổ gradient
        nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 6. Soft/Hard Update Target Network
        self.update_count += 1
        if self.update_count % TARGET_UPDATE_INTERVAL == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path: str):
        torch.save(self.q_network.state_dict(), path)

# ======================== TRAINING LOOP HELPER ==========================
# Hàm train() này thực ra không cần thiết nếu bạn dùng main_train.py 
# nhưng giữ lại để tương thích code cũ nếu cần
def train(agent: DQNAgent, env, num_episodes: int = 500):
    for episode in range(num_episodes):
        # Giả định env có hàm này (hoặc dùng logic trong main_train.py)
        if hasattr(env.state_builder, 'generate_initial_packet'):
             packet_data = env.state_builder.generate_initial_packet()
        else:
             # Fallback dummy packet
             packet_data = {"currentHoldingNodeId": "START", "stationDest": "END"}

        total_reward = env.simulate_episode(agent, packet_data, max_hops=10)

        if episode % 10 == 0:
            print(f"[Episode {episode:03d}] Total Reward = {total_reward:.3f}")