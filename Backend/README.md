# SAGIN Routing System với Reinforcement Learning

## 📋 Mục Lục
1. [Giới Thiệu Tổng Quan](#giới-thiệu-tổng-quan)
2. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
3. [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
4. [Cài Đặt và Cấu Hình](#cài-đặt-và-cấu-hình)
5. [Reinforcement Learning Agent](#reinforcement-learning-agent)
6. [API Endpoints](#api-endpoints)
7. [Simulation Scenarios](#simulation-scenarios)
8. [Training Model](#training-model)
9. [Kết Quả và Đánh Giá](#kết-quả-và-đánh-giá)
10. [Troubleshooting](#troubleshooting)

---

## 🌐 Giới Thiệu Tổng Quan

### Mục Tiêu Dự Án

Hệ thống **SAGIN (Space-Air-Ground Integrated Network) Routing** là một giải pháp định tuyến thông minh sử dụng **Deep Reinforcement Learning** để tối ưu hóa việc truyền dữ liệu trong mạng lưới tích hợp không gian-không trung-mặt đất.

### Vấn Đề Giải Quyết

1. **Định tuyến động (Dynamic Routing)**:
   - Mạng SAGIN có topology thay đổi liên tục do chuyển động của vệ tinh
   - Các thuật toán truyền thống (Dijkstra, Bellman-Ford) không tối ưu cho môi trường động
   
2. **Multi-Objective Optimization**:
   - Cân bằng giữa độ trễ (latency), khoảng cách (distance), và độ tin cậy (reliability)
   - Đáp ứng yêu cầu QoS (Quality of Service) khác nhau của từng loại dịch vụ

3. **Resource Management**:
   - Tối ưu hóa sử dụng tài nguyên (bandwidth, battery, processing power)
   - Phân bổ tải cân bằng giữa các nodes

### Đóng Góp Chính

- **Thuật toán Dueling DQN** cho routing động trong SAGIN
- **State representation** tối ưu cho network với 30+ nodes
- **Multi-scenario simulation** mô phỏng các điều kiện mạng khác nhau
- **REST API** đầy đủ cho tích hợp và monitoring
- **Web-based visualization** với Cesium 3D Globe

---

## Kiến Trúc Hệ Thống

### 1. **Agent: DuelingDQNAgent**
- **File**: `agent/dueling_dqn.py`
- **Chức năng**: Neural network agent học policy routing
- **Architecture**: 
  - Shared feature layers: [512, 256, 128] neurons
  - Value stream: Ước lượng V(s)
  - Advantage stream: Ước lượng A(s,a)
  - Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

### 2. **Environment: RoutingEnvironment**
- **File**: `environment/routing_env.py`
- **Chức năng**: Môi trường mô phỏng SAGIN network
- **Action space**: Discrete (chọn next node từ danh sách available nodes)
- **Observation space**: State vector với kích thước cố định

### 3. **State Builder: RoutingStateBuilder**
- **File**: `environment/state_builder.py`
- **Chức năng**: Xây dựng state vector từ network state
- **State dimension**: 
  - Node features: 30 nodes × 12 features = 360
  - Terminal features: 2 terminals × 6 features = 12
  - Global features: 8
  - **Tổng**: 380 dimensions

### 4. **Trainer: RoutingTrainer / EnhancedRoutingTrainer**
- **File**: `training/trainer.py`, `training/enhanced_trainer.py`
- **Chức năng**: Quản lý training loop, evaluation, checkpointing

---

## Thuật Toán và Công Thức

### 1. Dueling DQN Architecture

#### Công thức Q-value:
```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

Trong đó:
- **V(s)**: State value - giá trị của state s
- **A(s,a)**: Advantage - lợi thế của action a so với các actions khác
- **mean(A(s,a))**: Trung bình của tất cả advantages để đảm bảo identifiability

#### Lý do sử dụng Dueling DQN:
- Trong routing, nhiều actions có giá trị tương đương (ví dụ: chọn satellite A hay B khi cả hai đều tốt)
- Tách V(s) và A(s,a) giúp network học được rằng "state này tốt" mà không cần biết action cụ thể nào tốt nhất

### 2. Bellman Equation (DQN Update)

#### Target Q-value:
```
Q_target(s,a) = r + γ * max_a' Q_target(s', a')
```

#### Loss Function (Huber Loss):
```
L = smooth_l1_loss(Q_current(s,a) - Q_target(s,a))
```

Với **Double DQN** (để giảm overestimation):
```
Q_target(s,a) = r + γ * Q_target(s', argmax_a' Q_current(s', a'))
```

### 3. Experience Replay

#### Standard Replay Buffer:
- Lưu trữ experiences: (s, a, r, s', done)
- Sample ngẫu nhiên batch để train
- Giúp break correlation giữa consecutive experiences

#### Prioritized Experience Replay (Optional):
- Ưu tiên sample các experiences có TD-error cao
- Công thức priority:
```
priority = |TD_error|^α
```
- Importance sampling weights:
```
w_i = (N * P(i))^(-β) / max(w)
```

### 4. Epsilon-Greedy Exploration

#### Epsilon Decay:
```
ε(t) = max(ε_min, ε_start * decay^t)
```

Trong đó:
- `ε_start = 1.0`: Bắt đầu với 100% exploration
- `ε_min = 0.01`: Kết thúc với 1% exploration
- `decay = 0.9995`: Tốc độ giảm

### 5. Target Network Update

#### Hard Update (mỗi C steps):
```
θ_target ← θ_current
```

#### Soft Update (mỗi step):
```
θ_target ← τ * θ_current + (1 - τ) * θ_target
```

Với `τ = 0.005` (tau) để update mượt mà hơn.

### 6. Learning Starts (Warm-up Period) - CRITICAL

**Learning Starts** là số lượng experiences tối thiểu cần có trong replay buffer trước khi bắt đầu training. Đây là một **critical parameter** quan trọng.

#### Tại Sao Cần Learning Starts:

1. **Đảm Bảo Đa Dạng**: Cần đủ experiences đa dạng để học hiệu quả
2. **Tránh Overfitting**: Tránh học từ quá ít samples → overfitting
3. **Stable Training**: Đảm bảo replay buffer có đủ samples để sample batch

#### Công Thức:

```python
if len(replay_buffer) < learning_starts:
    return None  # Chưa train, chỉ collect experiences
```

#### Giá Trị Khuyến Nghị:

```yaml
rl_agent:
  dqn:
    learning_starts: 5000  # Tối thiểu 5000 experiences trước khi train
```

**Lưu ý Critical**:
- ⚠️ **Quá thấp** (< 1000): Model học từ quá ít samples → unstable, overfitting
- ⚠️ **Quá cao** (> 10000): Tốn thời gian chờ đợi, không cần thiết
- ✅ **Khuyến nghị**: 5000-10000 cho môi trường phức tạp như SAGIN

### 7. Reward Function

#### Các thành phần reward:

**1. Success Reward:**
```
R_success = 200.0 (nếu đến được destination)
```

**2. Progressive Rewards:**
```
R_progress = (distance_reduced / 100000) * progress_reward_scale
R_distance = -hop_distance / 10000000 * distance_reward_scale
```

**3. Node Quality Rewards:**
```
R_quality = node_quality_score * quality_reward_scale
R_satellite = +8.0 (nếu chọn satellite)
R_leo = +12.0 (nếu chọn LEO satellite)
```

**4. Resource Penalties:**
```
R_utilization = -penalty nếu utilization > threshold
R_battery = -penalty nếu battery < threshold
R_loss = -loss_rate * penalty_scale
```

**5. Path Efficiency:**
```
R_efficiency = +bonus nếu hops <= optimal_hops
R_inefficiency = -penalty nếu hops > optimal_hops
```

**6. QoS Compliance:**
```
R_qos = +30.0 nếu latency <= max_latency
R_qos = -15.0 nếu latency > max_latency
```

### 7. State Representation

#### Node Features (12 dimensions):
1. Resource utilization (0-1)
2. Packet buffer usage (0-1)
3. Packet loss rate (0-1)
4. Battery level (0-1)
5. Processing delay (0-1)
6. Bandwidth (0-1)
7. Is operational (0/1)
8. Is visited (0/1)
9. Distance to destination (normalized)
10. Distance to current node (normalized)
11. Node type encoding (0.2/0.5/0.8)
12. Quality score (0-1)

#### Terminal Features (6 dimensions):
1-3. Source terminal position (lat, lon, alt)
4-6. Destination terminal position (lat, lon, alt)

#### Global Features (8 dimensions):
1. Average network utilization
2. Average packet loss rate
3. Network congestion ratio
4. Operational nodes ratio
5. Current node utilization
6. Current node loss rate
7. Progress indicator (visited nodes ratio)
8. Scenario type (normal/congestion/failure)

---

## Cài Đặt và Chuẩn Bị

### 1. Yêu Cầu Hệ Thống

- **Python**: >= 3.8
- **PyTorch**: >= 2.1.0 (với CUDA nếu có GPU)
- **MongoDB**: Để lưu trữ network topology
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **GPU**: Không bắt buộc nhưng khuyến nghị (NVIDIA với CUDA)

### 2. Cài Đặt Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

### 3. Cấu Hình MongoDB

Đảm bảo MongoDB đang chạy và có dữ liệu:
- **Nodes**: Các nodes trong mạng (satellites, ground stations, etc.)
- **Terminals**: Các terminals cần routing

Kiểm tra kết nối:
```python
from models.database import db
db.connect()
nodes = list(db.get_collection('nodes').find({'isOperational': True}))
print(f"Found {len(nodes)} operational nodes")
```

### 4. Cấu Hình File Config

Chỉnh sửa `config.dev.yaml` hoặc `config.pro.yaml`:

```yaml
mongodb:
  uri: "mongodb://admin:password@localhost:27017/aiprancs?authSource=admin"
  database: "aiprancs"

rl_agent:
  dqn:
    learning_rate: 0.0001
    batch_size: 64
    buffer_size: 100000
    gamma: 0.99

training:
  max_episodes: 2000
  max_steps_per_episode: 15
  eval_frequency: 50
```

---

## Hướng Dẫn Training

### 1. Training Cơ Bản

#### Bước 1: Khởi động training
```bash
cd Backend
python -m training.train
```

#### Bước 2: Training với số episodes tùy chỉnh
```bash
python -m training.train --episodes 3000
```

#### Bước 3: Training với config file tùy chỉnh
```bash
python -m training.train --config custom_config.yaml
```

#### Bước 4: Resume từ checkpoint
```bash
python -m training.train --resume
```

### 2. Training với Enhanced Trainer

Enhanced Trainer bao gồm:
- **Curriculum Learning**: Từ dễ đến khó
- **Imitation Learning**: Học từ expert (Dijkstra)
- **Multi-objective Optimization**: Tối ưu nhiều mục tiêu

Để sử dụng Enhanced Trainer, set trong config:
```yaml
training:
  use_enhanced_trainer: true
```

Hoặc chỉnh trong code:
```python
from training.enhanced_trainer import EnhancedRoutingTrainer
trainer = EnhancedRoutingTrainer(config)
agent = trainer.train_from_database(num_episodes=2000)
```

### 3. Training Flow Chi Tiết

```
1. Initialize Environment
   ├── Load nodes từ MongoDB
   ├── Load terminals từ MongoDB
   └── Create RoutingEnvironment

2. Initialize Agent
   ├── Create DuelingDQN networks (Q-network + Target network)
   ├── Initialize Replay Buffer
   └── Set exploration parameters (epsilon)

3. Training Loop (cho mỗi episode):
   ├── Reset environment
   │   ├── Chọn random source & destination terminals
   │   └── Build initial state
   │
   ├── Episode Loop (cho mỗi step):
   │   ├── Select action (epsilon-greedy)
   │   ├── Execute action → (next_state, reward, done)
   │   ├── Store experience vào replay buffer
   │   ├── Sample batch từ replay buffer
   │   ├── Compute Q-targets
   │   ├── Update Q-network (backpropagation)
   │   ├── Update target network (mỗi C steps)
   │   └── Decay epsilon
   │
   ├── Evaluation (mỗi eval_frequency episodes):
   │   ├── Run evaluation episodes
   │   ├── Compute metrics (success rate, mean reward, etc.)
   │   └── Save best model nếu cải thiện
   │
   └── Checkpoint (mỗi save_frequency episodes):
       └── Save model checkpoint

4. Final Evaluation & Save
   ├── Run comprehensive evaluation
   └── Save final model
```

### 4. Monitoring Training

#### Tensorboard:
```bash
tensorboard --logdir=./logs/tensorboard
```

Mở browser tại `http://localhost:6006` để xem:
- Training reward
- Loss curves
- Success rate
- Epsilon decay
- Episode length

#### Log Files:
- `training.log`: Console logs
- `logs/tensorboard/`: Tensorboard events

---

## Cấu Hình và Tham Số

### ⚠️ CRITICAL PARAMETERS - Các Tham Số Quan Trọng Nhất

Trước khi đi vào chi tiết, đây là các tham số **CRITICAL** (quan trọng nhất) mà bạn **PHẢI** hiểu và tune đúng:

#### 1. **Learning Rate** (CRITICAL ⚠️)
- **Ảnh hưởng**: Quyết định tốc độ học và stability
- **Giá trị**: `0.0001` (default)
- **Quá cao** (> 0.001): Training không ổn định, loss explode, NaN
- **Quá thấp** (< 0.00001): Học quá chậm, không hội tụ
- **Khuyến nghị**: Bắt đầu với `0.0001`, tune trong khoảng `[0.00005, 0.0005]`

#### 2. **Gamma (Discount Factor)** (CRITICAL ⚠️)
- **Ảnh hưởng**: Quyết định agent quan tâm rewards xa đến đâu
- **Giá trị**: `0.99` (default)
- **Quá cao** (> 0.99): Agent quan tâm rewards quá xa → chậm học
- **Quá thấp** (< 0.9): Agent chỉ quan tâm immediate rewards → không tối ưu long-term
- **Khuyến nghị**: `0.95 - 0.99` cho episodic tasks như routing

#### 3. **Batch Size** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability và memory usage
- **Giá trị**: `64` (default)
- **Quá nhỏ** (< 32): Unstable gradients, noisy updates
- **Quá lớn** (> 256): Tốn memory, chậm training, có thể overfit
- **Khuyến nghị**: `32-128` tùy vào GPU memory

#### 4. **Replay Buffer Size** (CRITICAL ⚠️)
- **Ảnh hưởng**: Đa dạng của training data
- **Giá trị**: `100000` (default)
- **Quá nhỏ** (< 10000): Không đủ đa dạng, overfitting
- **Quá lớn** (> 1000000): Tốn memory, chậm sampling
- **Khuyến nghị**: `50000-200000` cho môi trường phức tạp

#### 5. **Target Update Frequency** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability của target Q-values
- **Giá trị**: `1000` steps (default)
- **Quá thường xuyên** (< 100): Target network thay đổi quá nhanh → unstable
- **Quá ít** (> 5000): Target network quá cũ → chậm học
- **Khuyến nghị**: `500-2000` steps

#### 6. **Epsilon Decay** (CRITICAL ⚠️)
- **Ảnh hưởng**: Cân bằng exploration vs exploitation
- **Giá trị**: `0.9995` (default)
- **Quá nhanh** (> 0.9999): Không explore đủ → stuck ở local optimum
- **Quá chậm** (< 0.99): Explore quá nhiều → không exploit knowledge
- **Khuyến nghị**: `0.999-0.9999` tùy vào số episodes

#### 7. **Reward Scale** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability và tốc độ học
- **Giá trị**: `success_reward: 200.0` (default)
- **Quá lớn** (> 1000): Q-values explode, training unstable
- **Quá nhỏ** (< 10): Agent không học được (rewards quá nhỏ so với noise)
- **Khuyến nghị**: Giữ rewards trong khoảng `[-100, 500]` để stable

#### 8. **State Dimension** (CRITICAL ⚠️)
- **Ảnh hưởng**: Complexity và training time
- **Giá trị**: `380` dimensions (default)
- **Quá lớn** (> 1000): Training chậm, cần nhiều data
- **Quá nhỏ** (< 100): Mất thông tin quan trọng
- **Khuyến nghị**: Giữ trong khoảng `200-500` cho routing

### 1. Hyperparameters Quan Trọng (Chi Tiết)

#### Learning Rate (CRITICAL):
```yaml
rl_agent:
  dqn:
    learning_rate: 0.0001  # Thấp hơn = stable hơn nhưng chậm hơn
```

#### Batch Size (CRITICAL):
```yaml
rl_agent:
  dqn:
    batch_size: 64  # Lớn hơn = stable hơn nhưng tốn memory
```
**⚠️ CRITICAL**: 
- Quá nhỏ (< 32): Unstable, noisy gradients
- Quá lớn (> 256): Tốn memory, có thể overfit
- **Khuyến nghị**: 64-128 cho GPU, 32-64 cho CPU

#### Buffer Size (CRITICAL):
```yaml
rl_agent:
  dqn:
    buffer_size: 100000  # Lưu trữ 100k experiences
```
**⚠️ CRITICAL**:
- Quá nhỏ (< 10000): Không đủ đa dạng, overfitting
- Quá lớn (> 1000000): Tốn memory, chậm sampling
- **Khuyến nghị**: 50000-200000 cho môi trường phức tạp

#### Gamma (Discount Factor) (CRITICAL):
```yaml
rl_agent:
  dqn:
    gamma: 0.99  # Gần 1.0 = quan tâm rewards xa hơn
```
**⚠️ CRITICAL**:
- Quá cao (> 0.99): Agent quan tâm rewards quá xa → chậm học
- Quá thấp (< 0.9): Agent chỉ quan tâm immediate rewards
- **Khuyến nghị**: 0.95-0.99 cho episodic tasks

#### Target Update Frequency (CRITICAL):
```yaml
rl_agent:
  dqn:
    target_update_interval: 1000  # Update target network mỗi 1000 steps
```
**⚠️ CRITICAL**:
- Quá thường xuyên (< 100): Target thay đổi quá nhanh → unstable
- Quá ít (> 5000): Target quá cũ → chậm học
- **Khuyến nghị**: 500-2000 steps

#### Learning Starts (CRITICAL):
```yaml
rl_agent:
  dqn:
    learning_starts: 5000  # Tối thiểu 5000 experiences trước khi train
```
**⚠️ CRITICAL**:
- Quá thấp (< 1000): Học từ quá ít samples → unstable, overfitting
- Quá cao (> 10000): Tốn thời gian chờ đợi không cần thiết
- **Khuyến nghị**: 5000-10000 cho môi trường phức tạp

### 2. Exploration Parameters

```yaml
rl_agent:
  dqn:
    exploration_initial_eps: 1.0      # Bắt đầu với 100% exploration
    exploration_final_eps: 0.01        # Kết thúc với 1% exploration
    exploration_decay: 0.9995          # Tốc độ decay
```

### 3. Network Architecture

```yaml
rl_agent:
  dqn:
    dueling:
      hidden_dims: [512, 256, 128]    # Kích thước các layers
      activation_fn: "elu"             # ELU tốt hơn ReLU cho DQN
      dropout_rate: 0.1                # Regularization
      use_layer_norm: true            # Training stability
```

### 4. Training Parameters

```yaml
training:
  max_episodes: 2000                  # Tổng số episodes
  max_steps_per_episode: 15           # Max steps mỗi episode
  eval_frequency: 50                   # Evaluate mỗi 50 episodes
  eval_episodes: 20                   # Số episodes để evaluate
  save_frequency: 100                 # Save checkpoint mỗi 100 episodes
  early_stopping_patience: 50          # Early stop nếu không cải thiện 50 evals
```

### 5. Reward Tuning (CRITICAL)

**⚠️ CRITICAL**: Reward engineering là một trong những phần quan trọng nhất của RL. Rewards quyết định agent học gì và học như thế nào.

```yaml
reward:
  success_reward: 200.0               # Reward khi thành công (CRITICAL)
  failure_penalty: -10.0               # Penalty khi thất bại
  step_penalty: -1.0                   # Penalty mỗi step
  hop_penalty: -2.0                    # Penalty mỗi hop
  progress_reward_scale: 100.0         # Scale cho progress reward (CRITICAL)
  proximity_bonus_scale: 50.0          # Bonus khi đến gần destination
```

#### Các Nguyên Tắc Quan Trọng:

1. **Reward Scale Balance**:
   - Success reward phải đủ lớn để "pay off" cho việc hoàn thành task
   - Nhưng không quá lớn để tránh Q-values explode
   - **Khuyến nghị**: `success_reward / |failure_penalty| ≈ 10-20`

2. **Shaped Rewards**:
   - Thêm intermediate rewards (progress, proximity) để guide learning
   - Giúp agent học nhanh hơn thay vì chỉ nhận reward ở cuối
   - **Khuyến nghị**: `progress_reward_scale` nên lớn hơn `step_penalty`

3. **Penalty Balance**:
   - Penalties không nên quá lớn để agent không sợ explore
   - Nhưng đủ lớn để discourage bad behaviors
   - **Khuyến nghị**: `|penalty| < success_reward / 10`

4. **Reward Normalization**:
   - Nếu rewards quá lớn (> 1000), normalize về khoảng [-100, 500]
   - Nếu rewards quá nhỏ (< 1), scale lên để agent có thể học được

---

## Tính Năng Nâng Cao

### 1. Curriculum Learning

**Mục đích**: Train từ scenarios đơn giản đến phức tạp

**Các Levels:**
- **Level 0 (Beginner)**: Gần (<1000km), ít nodes (5-30)
- **Level 1 (Easy)**: Gần (<2000km), nhiều nodes hơn (10-40)
- **Level 2 (Medium)**: Xa (<5000km), nhiều nodes (20-60)
- **Level 3 (Hard)**: Rất xa (<10000km), nhiều nodes (40-77), có QoS
- **Level 4 (Expert)**: Toàn cầu (<20000km), tất cả nodes (60-81)
- **Level 5 (Master)**: Không giới hạn

**Cấu hình:**
```yaml
curriculum:
  enabled: true
  min_success_rate: 0.7              # Advance khi success rate >= 70%
  min_episodes_at_level: 100         # Tối thiểu 100 episodes mỗi level
  adaptive: true                      # Adaptive difficulty
```

### 2. Imitation Learning

**Mục đích**: Học từ expert demonstrations (Dijkstra algorithm)

**Phương pháp**: DAGGER (Dataset Aggregation)
- Bắt đầu với 100% expert actions
- Gradually giảm expert ratio khi agent cải thiện
- Mix expert actions với agent actions

**Cấu hình:**
```yaml
imitation_learning:
  enabled: true
  use_dagger: true
  expert_ratio: 0.3                   # 30% expert actions ban đầu
  bc_loss_weight: 0.5                # Behavior Cloning loss weight
```

### 3. Multi-Objective Optimization

**Mục đích**: Tối ưu đồng thời nhiều mục tiêu (latency, reliability, energy)

**Phương pháp**: Pareto Front
- Tìm các solutions không bị dominate bởi solutions khác
- User có thể chọn solution dựa trên preference

**Cấu hình:**
```yaml
multi_objective:
  enabled: true
  use_pareto: true
  pareto_front_size: 10
  latency_weight: 0.4
  reliability_weight: 0.3
  energy_weight: 0.3
  adaptive_weights: true              # Tự động điều chỉnh weights
```

### 4. Prioritized Experience Replay

**Mục đích**: Ưu tiên học từ các experiences quan trọng (TD-error cao)

**Cấu hình:**
```yaml
rl_agent:
  dqn:
    use_prioritized_replay: true
```

**Công thức:**
```
priority = |TD_error|^α
P(i) = priority_i / Σ priority_j
w_i = (N * P(i))^(-β)
```

### 5. Double DQN

**Mục đích**: Giảm overestimation của Q-values

**Cấu hình:**
```yaml
rl_agent:
  dqn:
    use_double_dqn: true
```

**Công thức:**
```
Q_target = r + γ * Q_target(s', argmax_a' Q_current(s', a'))
```

---

## Monitoring và Đánh Giá

### 1. Metrics Được Track

#### Training Metrics:
- **Episode Reward**: Tổng reward mỗi episode
- **Episode Length**: Số steps mỗi episode
- **Loss**: Training loss (Huber loss)
- **Q-values**: Mean Q-values
- **Epsilon**: Exploration rate
- **Success Rate**: Tỷ lệ episodes thành công

#### Evaluation Metrics:
- **Mean Reward**: Reward trung bình
- **Success Rate**: Tỷ lệ thành công
- **Mean Hops**: Số hops trung bình
- **Mean Latency**: Latency trung bình (ms)
- **Mean Distance**: Khoảng cách trung bình (km)

### 2. Model Checkpoints

#### Best Model:
- **Path**: `models/best_models/best_model.pt`
- **Saved khi**: Evaluation reward cải thiện

#### Checkpoints:
- **Path**: `models/checkpoints/checkpoint_ep{episode}.pt`
- **Saved mỗi**: `save_frequency` episodes

#### Final Model:
- **Path**: `models/rl_agent/final_model.pt`
- **Saved khi**: Training hoàn thành

### 3. Evaluation Script

```python
from training.trainer import RoutingTrainer
from models.database import db

# Load model
trainer = RoutingTrainer(config)
metrics = trainer.load_and_evaluate(
    model_path='./models/best_models/best_model.pt',
    nodes=nodes,
    terminals=terminals,
    num_episodes=50
)

print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Mean Hops: {metrics['mean_hops']:.1f}")
print(f"Mean Latency: {metrics['mean_latency']:.2f}ms")
```

### 4. So Sánh với Baseline

Hệ thống có thể so sánh với:
- **Dijkstra**: Shortest path algorithm
- **Heuristic**: Rule-based routing

```python
# Trong evaluation, so sánh với Dijkstra
enable_dijkstra_comparison: true
```

---

## Troubleshooting

### 1. Training Không Hội Tụ

**Triệu chứng**: Reward không tăng, loss không giảm

**Giải pháp**:
- Giảm learning rate: `0.0001 → 0.00005`
- Tăng batch size: `64 → 128`
- Kiểm tra reward scale (có thể quá lớn/nhỏ)
- Tăng exploration: `exploration_final_eps: 0.05`
- Kiểm tra state normalization

### 2. Out of Memory

**Triệu chứng**: CUDA out of memory

**Giải pháp**:
- Giảm batch size: `64 → 32`
- Giảm buffer size: `100000 → 50000`
- Giảm max_nodes trong state: `30 → 20`
- Sử dụng CPU thay vì GPU

### 3. Success Rate Thấp

**Triệu chứng**: Agent không tìm được đường đi

**Giải pháp**:
- Tăng success reward: `200.0 → 500.0`
- Giảm failure penalty: `-10.0 → -5.0`
- Tăng progress reward scale: `100.0 → 200.0`
- Sử dụng Curriculum Learning
- Sử dụng Imitation Learning để bootstrap

### 4. Training Quá Chậm

**Triệu chứng**: Mỗi episode mất quá nhiều thời gian

**Giải pháp**:
- Giảm max_steps_per_episode: `15 → 10`
- Giảm max_nodes: `30 → 20`
- Tắt một số features không cần thiết
- Sử dụng GPU nếu có
- Tăng batch size để train hiệu quả hơn

### 5. Model Không Load Được

**Triệu chứng**: Lỗi khi load checkpoint

**Giải pháp**:
- Kiểm tra action_dim có khớp không
- Kiểm tra state_dim có khớp không
- Load với `strict=False` nếu architecture thay đổi
- Kiểm tra PyTorch version compatibility

### 6. NaN Loss

**Triệu chứng**: Loss = NaN

**Giải pháp**:
- Kiểm tra reward scale (có thể quá lớn)
- Thêm gradient clipping: `gradient_clip: 10.0`
- Kiểm tra state có NaN/Inf không
- Normalize rewards: `reward = reward / 100.0`

---

## Best Practices

### 1. Hyperparameter Tuning

- **Bắt đầu với default values** trong config
- **Tune từng tham số một** để hiểu ảnh hưởng
- **Sử dụng Tensorboard** để visualize
- **Early stopping** để tránh overfitting

### 2. Reward Engineering

- **Reward scale quan trọng**: Quá lớn → unstable, quá nhỏ → chậm học
- **Shaped rewards**: Thêm intermediate rewards để guide learning
- **Penalty balance**: Không penalty quá mạnh → agent không dám explore

### 3. State Design

- **Normalize features**: Tất cả features về [0, 1] hoặc [-1, 1]
- **Feature selection**: Chỉ giữ features quan trọng
- **Caching**: Cache các tính toán tốn kém (distance, quality)

### 4. Training Strategy

- **Warm-up**: Cho agent explore nhiều trước khi train (`learning_starts: 5000`)
- **Curriculum**: Bắt đầu từ scenarios đơn giản
- **Evaluation**: Evaluate thường xuyên nhưng không quá nhiều (tốn thời gian)

### 5. Model Selection

- **Best model**: Chọn model có evaluation reward cao nhất
- **Ensemble**: Có thể ensemble nhiều models để tăng robustness
- **Transfer learning**: Fine-tune từ pretrained model

---

## Tài Liệu Tham Khảo

### Papers:
1. **Dueling DQN**: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
2. **Double DQN**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
3. **Prioritized Replay**: "Prioritized Experience Replay" (Schaul et al., 2016)
4. **Curriculum Learning**: "Curriculum Learning" (Bengio et al., 2009)
5. **DAGGER**: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (Ross et al., 2011)

### Code Structure:
- `agent/dueling_dqn.py`: Dueling DQN implementation
- `environment/routing_env.py`: Environment implementation
- `environment/state_builder.py`: State representation
- `training/trainer.py`: Standard trainer
- `training/enhanced_trainer.py`: Enhanced trainer với advanced features
- `training/curriculum_learning.py`: Curriculum learning
- `training/imitation_learning.py`: Imitation learning
- `training/multi_objective.py`: Multi-objective optimization

---

## Liên Hệ và Hỗ Trợ

Nếu có vấn đề hoặc câu hỏi, vui lòng:
1. Kiểm tra logs trong `training.log`
2. Xem Tensorboard để visualize training
3. Kiểm tra config file có đúng không
4. Đọc troubleshooting section ở trên

---

**Chúc bạn training thành công! 🚀**

