# Quick Start Guide

Get started with SAGIN RL routing in 5 minutes!

## Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /Users/anhnon/PBL4/src/reinforcement

# Install requirements
pip install -r requirements.txt
```

## Step 2: Start MongoDB

```bash
# Start MongoDB (choose one method)

# Method 1: Direct command
mongod --dbpath ~/data/db

# Method 2: macOS with Homebrew
brew services start mongodb-community

# Method 3: Linux with systemd
sudo systemctl start mongod

# Method 4: Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## Step 3: Train Your First Model

### Option A: Using main.py (Simple)

```bash
# Basic training with default config
python main.py --mode train

# Training with specific config
python main.py --mode train --config configs/dynamic_config.yaml
```

### Option B: Using training script (Advanced)

```bash
# Training with more options
python scripts/train.py --config configs/dynamic_config.yaml --episodes 1000

# With custom name
python scripts/train.py --config configs/dynamic_config.yaml --name my_first_model
```

### What to Expect

Training will show real-time progress:

```
Loading configuration from: configs/dynamic_config.yaml

======================================================================
TRAINING CONFIGURATION
======================================================================
Experiment Name: train_20241124_143022
Config File: configs/dynamic_config.yaml
Episodes: 2000
Max Hops: 12
Learning Rate: 0.0005
Batch Size: 128
Device: auto

Dynamics:
  Weather: True
  Traffic: True
  Failures: True
  Mobility: True
======================================================================

Initializing trainer...
Starting dynamic SAGIN RL training...

[Episode 000] Reward: 342.15 | Delivered: Yes | Latency: 345.67ms | Hops: 5
[Episode 010] Reward: 456.23 | Delivered: Yes | Latency: 289.12ms | Hops: 4
[Episode 020] Reward: 534.89 | Delivered: Yes | Latency: 245.34ms | Hops: 4
...

--- Evaluating agent at episode 100 ---
Evaluation Results:
  - Average Reward: 678.45
  - Packet Delivery Rate: 82.00%
  - Average Latency (for delivered packets): 198.56ms
  - Average Hops (for delivered packets): 3.80
-------------------------------------------------
```

### Checkpoints

Models are automatically saved to:
```
checkpoints/models/agent_episode_100.pth
checkpoints/models/agent_episode_200.pth
checkpoints/models/agent_episode_300.pth
...
```

## Step 4: Test Your Model

### Option A: Quick Evaluation

```bash
# Evaluate latest checkpoint
python main.py --mode eval --config configs/dynamic_config.yaml
```

### Option B: Detailed Testing

```bash
# Test specific checkpoint
python scripts/test.py --checkpoint checkpoints/models/agent_episode_1000.pth --episodes 100

# Save results to file
python scripts/test.py \
    --checkpoint checkpoints/models/agent_episode_1000.pth \
    --episodes 100 \
    --output results/eval_results.json
```

### Expected Output

```
======================================================================
TEST CONFIGURATION
======================================================================
Checkpoint: checkpoints/models/agent_episode_1000.pth
Config File: configs/dynamic_config.yaml
Episodes: 100
Deterministic: False
======================================================================

Initializing environment and agent...
Loaded 19 nodes from database
Loading model from checkpoint...
Checkpoint loaded successfully!

Evaluating agent for 100 episodes...
  Progress: 10/100 episodes completed
  Progress: 20/100 episodes completed
  ...
  Progress: 100/100 episodes completed

======================================================================
EVALUATION RESULTS
======================================================================
Total Episodes:            100

Average Reward:            756.32 ± 123.45

Packet Delivery Rate:      85.00%
Failed Deliveries:         15

Average Latency:           187.45 ± 45.23 ms
Average Hops:              3.65 ± 0.89
======================================================================
```

## Common Issues & Solutions

### Issue 1: MongoDB Connection Error

```
Error: Could not connect to MongoDB
```

**Solution:**
```bash
# Check if MongoDB is running
ps aux | grep mongod

# Start MongoDB if not running
mongod --dbpath ~/data/db
```

### Issue 2: No Nodes in Database

```
WARNING: No nodes found in database!
```

**Solution:** You need to populate the database first. The training will need initial network data.

### Issue 3: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
python scripts/train.py --config configs/dynamic_config.yaml --device cpu

# Or reduce batch size in config
# Edit configs/dynamic_config.yaml:
# training:
#   batch_size: 64  # Reduced from 128
```

### Issue 4: Import Errors

```
ModuleNotFoundError: No module named 'agents'
```

**Solution:**
```bash
# Run from project root
cd /Users/anhnon/PBL4/src/reinforcement
python main.py --mode train

# Or add to PYTHONPATH
export PYTHONPATH=/Users/anhnon/PBL4/src/reinforcement:$PYTHONPATH
```

## Next Steps

1. **Experiment with different scenarios:**
   - Edit configs to enable/disable dynamics
   - Try baseline, dynamic, and stress test scenarios

2. **Tune hyperparameters:**
   - Adjust learning rate, batch size, epsilon decay
   - Modify reward weights

3. **Analyze results:**
   - Compare different checkpoints
   - Visualize training curves
   - Test under various network conditions

4. **Read full documentation:**
   - [README.md](README.md) - Complete project overview
   - [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training guide

## Training Tips

### 1. Start Simple

Begin with baseline scenario (no dynamics):
```bash
# Edit configs/base_config.yaml to disable dynamics
python main.py --mode train --config configs/base_config.yaml
```

### 2. Monitor Progress

Watch for:
- **Increasing rewards** - Agent is improving
- **High delivery rate** - Good routing decisions
- **Decreasing latency** - More efficient paths
- **Stable hop count** - Consistent routing

### 3. Save Regularly

Checkpoints are saved automatically, but you can also:
- Use Ctrl+C to interrupt and save
- Keep multiple checkpoints for comparison

### 4. Evaluate Often

Test your model every 100-200 episodes to track improvement.

## Example Training Session

```bash
# 1. Start MongoDB
brew services start mongodb-community

# 2. Train model
python scripts/train.py \
    --config configs/dynamic_config.yaml \
    --episodes 2000 \
    --name experiment_001

# 3. Wait for training (or monitor in another terminal)
# Training will take ~1-3 hours depending on hardware

# 4. Test the best checkpoint
python scripts/test.py \
    --checkpoint checkpoints/models/agent_episode_2000.pth \
    --episodes 200 \
    --output results/experiment_001_results.json

# 5. Compare with earlier checkpoints
python scripts/test.py \
    --checkpoint checkpoints/models/agent_episode_1000.pth \
    --episodes 200 \
    --output results/experiment_001_mid_results.json
```

That's it! You're now ready to train and test your SAGIN RL routing agent! 🚀
