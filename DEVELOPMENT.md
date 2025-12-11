# SAGIN Routing System - Development Guide

## 📋 Table of Contents
1. [Training Guide](#training-guide)
2. [Configuration and Parameters](#configuration-and-parameters)
3. [Advanced Features](#advanced-features)
4. [Monitoring and Evaluation](#monitoring-and-evaluation)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Training Guide

### 1. Basic Training

#### Step 1: Start training
```bash
cd Backend
python -m training.train
```

#### Step 2: Training with custom episodes
```bash
python -m training.train --episodes 3000
```

#### Step 3: Training with custom config file
```bash
python -m training.train --config custom_config.yaml
```

#### Step 4: Resume from checkpoint
```bash
python -m training.train --resume
```

### 2. Training with Enhanced Trainer

Enhanced Trainer includes:
- **Curriculum Learning**: Train from easy to hard
- **Imitation Learning**: Learn from expert (Dijkstra)
- **Multi-objective Optimization**: Optimize multiple objectives

To use Enhanced Trainer, set in config:
```yaml
training:
  use_enhanced_trainer: true
```

### 3. Detailed Training Flow

```
1. Initialize Environment
   ├── Load nodes from MongoDB
   ├── Load terminals from MongoDB
   └── Create RoutingEnvironment

2. Initialize Agent
   ├── Create DuelingDQN networks (Q-network + Target network)
   ├── Initialize Replay Buffer
   └── Set exploration parameters (epsilon)

3. Training Loop (for each episode):
   ├── Reset environment
   │   ├── Select random source & destination terminals
   │   └── Build initial state
   │
   ├── Episode Loop (for each step):
   │   ├── Select action (epsilon-greedy)
   │   ├── Execute action → (next_state, reward, done)
   │   ├── Store experience in replay buffer
   │   ├── Sample batch from replay buffer
   │   ├── Compute Q-targets
   │   ├── Update Q-network (backpropagation)
   │   ├── Update target network (every C steps)
   │   └── Decay epsilon
   │
   ├── Evaluation (every eval_frequency episodes):
   │   ├── Run evaluation episodes
   │   ├── Compute metrics (success rate, mean reward, etc.)
   │   └── Save best model if improved
   │
   └── Checkpoint (every save_frequency episodes):
       └── Save model checkpoint

4. Final Evaluation & Save
   ├── Run comprehensive evaluation
   └── Save final model
```

---

## Configuration and Parameters

### ⚠️ CRITICAL PARAMETERS - Most Important Parameters

#### 1. **Learning Rate** (CRITICAL ⚠️)
- **Impact**: Determines learning speed and stability
- **Value**: `0.0001` (default)
- **Too high** (> 0.001): Unstable training, loss explodes, NaN
- **Too low** (< 0.00001): Learns too slowly, doesn't converge
- **Recommendation**: Start with `0.0001`, tune in range `[0.00005, 0.0005]`

```yaml
rl_agent:
  dqn:
    learning_rate: 0.0001
```

#### 2. **Gamma (Discount Factor)** (CRITICAL ⚠️)
- **Impact**: Determines how far ahead agent considers rewards
- **Value**: `0.99` (default)
- **Too high** (> 0.99): Agent considers rewards too far ahead → slow learning
- **Too low** (< 0.9): Agent only considers immediate rewards → not optimal long-term
- **Recommendation**: `0.95 - 0.99` for episodic tasks like routing

```yaml
rl_agent:
  dqn:
    gamma: 0.99
```

#### 3. **Batch Size** (CRITICAL ⚠️)
- **Impact**: Stability and memory usage
- **Value**: `64` (default)
- **Too small** (< 32): Unstable gradients, noisy updates
- **Too large** (> 256): High memory usage, slow training, may overfit
- **Recommendation**: `32-128` depending on GPU memory

```yaml
rl_agent:
  dqn:
    batch_size: 64
```

#### 4. **Replay Buffer Size** (CRITICAL ⚠️)
- **Impact**: Diversity of training data
- **Value**: `100000` (default)
- **Too small** (< 10000): Not enough diversity, overfitting
- **Too large** (> 1000000): High memory usage, slow sampling
- **Recommendation**: `50000-200000` for complex environments

```yaml
rl_agent:
  dqn:
    buffer_size: 100000
```

#### 5. **Target Update Frequency** (CRITICAL ⚠️)
- **Impact**: Stability of target Q-values
- **Value**: `1000` steps (default)
- **Too frequent** (< 100): Target network changes too fast → unstable
- **Too rare** (> 5000): Target network too old → slow learning
- **Recommendation**: `500-2000` steps

```yaml
rl_agent:
  dqn:
    target_update_interval: 1000
```

#### 6. **Learning Starts** (CRITICAL ⚠️)
- **Impact**: Ensures enough experiences before training
- **Value**: `5000` (default)
- **Too low** (< 1000): Learning from too few samples → unstable, overfitting
- **Too high** (> 10000): Wasting time waiting unnecessarily
- **Recommendation**: `5000-10000` for complex environments

```yaml
rl_agent:
  dqn:
    learning_starts: 5000
```

#### 7. **Reward Scale** (CRITICAL ⚠️)
- **Impact**: Stability and learning speed
- **Value**: `success_reward: 200.0` (default)
- **Too large** (> 1000): Q-values explode, training unstable
- **Too small** (< 10): Agent can't learn (rewards too small compared to noise)
- **Recommendation**: Keep rewards in range `[-100, 500]` for stability

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
    exploration_initial_eps: 1.0      # Start with 100% exploration
    exploration_final_eps: 0.01        # End with 1% exploration
    exploration_decay: 0.9995          # Decay rate
```

### Network Architecture

```yaml
rl_agent:
  dqn:
    dueling:
      hidden_dims: [512, 256, 128]    # Layer sizes
      activation_fn: "elu"             # ELU better than ReLU for DQN
      dropout_rate: 0.1                # Regularization
      use_layer_norm: true            # Training stability
```

### Training Parameters

```yaml
training:
  max_episodes: 2000                  # Total episodes
  max_steps_per_episode: 15           # Max steps per episode
  eval_frequency: 50                   # Evaluate every 50 episodes
  eval_episodes: 20                   # Number of episodes to evaluate
  save_frequency: 100                 # Save checkpoint every 100 episodes
  early_stopping_patience: 50          # Early stop if no improvement for 50 evals
```

---

## Advanced Features

### 1. Curriculum Learning

**Purpose**: Train from simple to complex scenarios

**Levels:**
- **Level 0 (Beginner)**: Close (<1000km), few nodes (5-30)
- **Level 1 (Easy)**: Close (<2000km), more nodes (10-40)
- **Level 2 (Medium)**: Far (<5000km), many nodes (20-60)
- **Level 3 (Hard)**: Very far (<10000km), many nodes (40-77), with QoS
- **Level 4 (Expert)**: Global (<20000km), all nodes (60-81)
- **Level 5 (Master)**: No limits

**Configuration:**
```yaml
curriculum:
  enabled: true
  min_success_rate: 0.7              # Advance when success rate >= 70%
  min_episodes_at_level: 100         # Minimum 100 episodes per level
  adaptive: true                      # Adaptive difficulty
```

### 2. Imitation Learning

**Purpose**: Learn from expert demonstrations (Dijkstra algorithm)

**Method**: DAGGER (Dataset Aggregation)
- Start with 100% expert actions
- Gradually reduce expert ratio as agent improves
- Mix expert actions with agent actions

**Configuration:**
```yaml
imitation_learning:
  enabled: true
  use_dagger: true
  expert_ratio: 0.3                   # 30% expert actions initially
  bc_loss_weight: 0.5                # Behavior Cloning loss weight
```

### 3. Multi-Objective Optimization

**Purpose**: Optimize multiple objectives simultaneously (latency, reliability, energy)

**Configuration:**
```yaml
multi_objective:
  enabled: true
  use_pareto: true
  pareto_front_size: 10
  latency_weight: 0.4
  reliability_weight: 0.3
  energy_weight: 0.3
  adaptive_weights: true              # Automatically adjust weights
```

---

## Monitoring and Evaluation

### 1. Tensorboard

```bash
tensorboard --logdir=./logs/tensorboard
```

Open browser at `http://localhost:6006` to view:
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

### 1. Training Not Converging

**Symptoms**: Reward not increasing, loss not decreasing

**Solutions**:
- Reduce learning rate: `0.0001 → 0.00005`
- Increase batch size: `64 → 128`
- Check reward scale (may be too large/small)
- Increase exploration: `exploration_final_eps: 0.05`
- Check state normalization

### 2. Out of Memory

**Symptoms**: CUDA out of memory

**Solutions**:
- Reduce batch size: `64 → 32`
- Reduce buffer size: `100000 → 50000`
- Reduce max_nodes in state: `30 → 20`
- Use CPU instead of GPU

### 3. Low Success Rate

**Symptoms**: Agent cannot find paths

**Solutions**:
- Increase success reward: `200.0 → 500.0`
- Reduce failure penalty: `-10.0 → -5.0`
- Increase progress reward scale: `100.0 → 200.0`
- Use Curriculum Learning
- Use Imitation Learning to bootstrap

### 4. Training Too Slow

**Symptoms**: Each episode takes too long

**Solutions**:
- Reduce max_steps_per_episode: `15 → 10`
- Reduce max_nodes: `30 → 20`
- Disable unnecessary features
- Use GPU if available
- Increase batch size for more efficient training

### 5. NaN Loss

**Symptoms**: Loss = NaN

**Solutions**:
- Check reward scale (may be too large)
- Add gradient clipping: `gradient_clip: 10.0`
- Check for NaN/Inf in state
- Normalize rewards: `reward = reward / 100.0`

---

## Best Practices

### 1. Hyperparameter Tuning

- **Start with default values** in config
- **Tune one parameter at a time** to understand impact
- **Use Tensorboard** to visualize
- **Early stopping** to avoid overfitting

### 2. Reward Engineering

- **Reward scale is important**: Too large → unstable, too small → slow learning
- **Shaped rewards**: Add intermediate rewards to guide learning
- **Penalty balance**: Don't penalize too harshly → agent won't dare explore

### 3. State Design

- **Normalize features**: All features to [0, 1] or [-1, 1]
- **Feature selection**: Keep only important features
- **Caching**: Cache expensive computations (distance, quality)

### 4. Training Strategy

- **Warm-up**: Let agent explore before training (`learning_starts: 5000`)
- **Curriculum**: Start from simple scenarios
- **Evaluation**: Evaluate frequently but not too often (time-consuming)

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
