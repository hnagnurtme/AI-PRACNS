# Training and Testing Guide

Complete guide for training and testing the SAGIN RL routing agent.

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB

```bash
# Start MongoDB service
mongod --dbpath /path/to/your/data

# Or using brew (macOS)
brew services start mongodb-community

# Or using systemctl (Linux)
sudo systemctl start mongod
```

### 3. Verify Installation

```bash
# Test MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017/').server_info())"

# Test PyTorch
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Training

### Basic Training

Train with default configuration:

```bash
python main.py --mode train
```

### Training with Specific Configuration

```bash
# Use baseline scenario (ideal conditions)
python main.py --mode train --config configs/base_config.yaml

# Use dynamic scenario (realistic conditions)
python main.py --mode train --config configs/dynamic_config.yaml
```

### Training Options

The training will:
- Train for the number of episodes specified in config (default: 1000-2000)
- Save checkpoints every 100 episodes to `checkpoints/models/`
- Log progress every 10 episodes
- Evaluate every 100 episodes
- Display metrics: reward, delivery rate, latency, hops

### Monitor Training Progress

Training logs are displayed in real-time:

```
[Episode 000] Reward: 450.23 | Delivered: Yes | Latency: 234.56ms | Hops: 4
[Episode 010] Reward: 523.45 | Delivered: Yes | Latency: 189.23ms | Hops: 3
[Episode 020] Reward: 612.34 | Delivered: Yes | Latency: 156.78ms | Hops: 3
...
[Episode 100] Reward: 845.67 | Delivered: Yes | Latency: 123.45ms | Hops: 3

--- Evaluating agent at episode 100 ---
Evaluation Results:
  - Average Reward: 789.45
  - Packet Delivery Rate: 87.50%
  - Average Latency (for delivered packets): 145.23ms
  - Average Hops (for delivered packets): 3.20
-------------------------------------------------
```

### Training with Different Scenarios

#### 1. Baseline Scenario (Easy)

```bash
# Edit configs/base_config.yaml to disable dynamics
python main.py --mode train --config configs/base_config.yaml
```

Perfect for:
- Initial testing
- Baseline performance measurement
- Debugging

#### 2. Dynamic Scenario (Normal)

```bash
python main.py --mode train --config configs/dynamic_config.yaml
```

Includes:
- Weather effects
- Traffic variations
- Node mobility
- Occasional failures

#### 3. Stress Test Scenario (Hard)

```bash
# Create stress test config or modify dynamic_config.yaml
# Increase failure rates, enable severe weather
python main.py --mode train --config configs/dynamic_config.yaml
```

Most challenging:
- High failure rates
- Extreme weather
- Heavy traffic load
- Rapid topology changes

### Resume Training

To continue training from a checkpoint:

```python
# Modify training/train.py to load checkpoint
# Add this in DynamicTrainingManager.__init__():
if 'checkpoint_path' in self.config:
    self.agent.q_network.load_state_dict(
        torch.load(self.config['checkpoint_path'])
    )
```

Then run:

```bash
python main.py --mode train --config configs/resume_config.yaml
```

---

## Testing / Evaluation

### Basic Evaluation

```bash
python main.py --mode eval --config configs/dynamic_config.yaml
```

This will:
- Compare RL agent with baseline algorithms (Dijkstra, Random)
- Display comparative metrics
- Show algorithm comparison results

### Evaluate Specific Checkpoint

Create an evaluation script:

<function_calls>
<invoke name="Read">
<parameter name="file_path">/Users/anhnon/PBL4/src/reinforcement/scripts