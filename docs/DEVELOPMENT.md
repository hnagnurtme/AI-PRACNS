# SAGIN Routing System - Development Guide

## 📋 Mục Lục
1. [Hướng Dẫn Training](#hướng-dẫn-training)
2. [Cấu Hình và Tham Số](#cấu-hình-và-tham-số)
3. [Tính Năng Nâng Cao](#tính-năng-nâng-cao)
4. [Monitoring và Đánh Giá](#monitoring-và-đánh-giá)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

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

---

## Cấu Hình và Tham Số

### ⚠️ CRITICAL PARAMETERS - Các Tham Số Quan Trọng Nhất

#### 1. **Learning Rate** (CRITICAL ⚠️)
- **Ảnh hưởng**: Quyết định tốc độ học và stability
- **Giá trị**: `0.0001` (default)
- **Quá cao** (> 0.001): Training không ổn định, loss explode, NaN
- **Quá thấp** (< 0.00001): Học quá chậm, không hội tụ
- **Khuyến nghị**: Bắt đầu với `0.0001`, tune trong khoảng `[0.00005, 0.0005]`

```yaml
rl_agent:
  dqn:
    learning_rate: 0.0001
```

#### 2. **Gamma (Discount Factor)** (CRITICAL ⚠️)
- **Ảnh hưởng**: Quyết định agent quan tâm rewards xa đến đâu
- **Giá trị**: `0.99` (default)
- **Quá cao** (> 0.99): Agent quan tâm rewards quá xa → chậm học
- **Quá thấp** (< 0.9): Agent chỉ quan tâm immediate rewards → không tối ưu long-term
- **Khuyến nghị**: `0.95 - 0.99` cho episodic tasks như routing

```yaml
rl_agent:
  dqn:
    gamma: 0.99
```

#### 3. **Batch Size** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability và memory usage
- **Giá trị**: `64` (default)
- **Quá nhỏ** (< 32): Unstable gradients, noisy updates
- **Quá lớn** (> 256): Tốn memory, chậm training, có thể overfit
- **Khuyến nghị**: `32-128` tùy vào GPU memory

```yaml
rl_agent:
  dqn:
    batch_size: 64
```

#### 4. **Replay Buffer Size** (CRITICAL ⚠️)
- **Ảnh hưởng**: Đa dạng của training data
- **Giá trị**: `100000` (default)
- **Quá nhỏ** (< 10000): Không đủ đa dạng, overfitting
- **Quá lớn** (> 1000000): Tốn memory, chậm sampling
- **Khuyến nghị**: `50000-200000` cho môi trường phức tạp

```yaml
rl_agent:
  dqn:
    buffer_size: 100000
```

#### 5. **Target Update Frequency** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability của target Q-values
- **Giá trị**: `1000` steps (default)
- **Quá thường xuyên** (< 100): Target network thay đổi quá nhanh → unstable
- **Quá ít** (> 5000): Target network quá cũ → chậm học
- **Khuyến nghị**: `500-2000` steps

```yaml
rl_agent:
  dqn:
    target_update_interval: 1000
```

#### 6. **Learning Starts** (CRITICAL ⚠️)
- **Ảnh hưởng**: Đảm bảo đủ experiences trước khi train
- **Giá trị**: `5000` (default)
- **Quá thấp** (< 1000): Học từ quá ít samples → unstable, overfitting
- **Quá cao** (> 10000): Tốn thời gian chờ đợi không cần thiết
- **Khuyến nghị**: `5000-10000` cho môi trường phức tạp

```yaml
rl_agent:
  dqn:
    learning_starts: 5000
```

#### 7. **Reward Scale** (CRITICAL ⚠️)
- **Ảnh hưởng**: Stability và tốc độ học
- **Giá trị**: `success_reward: 200.0` (default)
- **Quá lớn** (> 1000): Q-values explode, training unstable
- **Quá nhỏ** (< 10): Agent không học được (rewards quá nhỏ so với noise)
- **Khuyến nghị**: Giữ rewards trong khoảng `[-100, 500]` để stable

```yaml
reward:
  success_reward: 200.0
  failure_penalty: -10.0
  progress_reward_scale: 100.0
```

### Exploration Parameters

```yaml
rl_agent:
  dqn:
    exploration_initial_eps: 1.0      # Bắt đầu với 100% exploration
    exploration_final_eps: 0.01        # Kết thúc với 1% exploration
    exploration_decay: 0.9995          # Tốc độ decay
```

### Network Architecture

```yaml
rl_agent:
  dqn:
    dueling:
      hidden_dims: [512, 256, 128]    # Kích thước các layers
      activation_fn: "elu"             # ELU tốt hơn ReLU cho DQN
      dropout_rate: 0.1                # Regularization
      use_layer_norm: true            # Training stability
```

### Training Parameters

```yaml
training:
  max_episodes: 2000                  # Tổng số episodes
  max_steps_per_episode: 15           # Max steps mỗi episode
  eval_frequency: 50                   # Evaluate mỗi 50 episodes
  eval_episodes: 20                   # Số episodes để evaluate
  save_frequency: 100                 # Save checkpoint mỗi 100 episodes
  early_stopping_patience: 50          # Early stop nếu không cải thiện 50 evals
```

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

---

## Monitoring và Đánh Giá

### 1. Tensorboard

```bash
tensorboard --logdir=./logs/tensorboard
```

Mở browser tại `http://localhost:6006` để xem:
- Training reward
- Loss curves
- Success rate
- Epsilon decay
- Episode length

### 2. Evaluation Script

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

### 5. NaN Loss

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
