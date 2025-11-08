# Testing Guide for RL Routing Optimization

## Overview
This guide explains how to test and validate the optimized RL routing algorithm against Dijkstra.

## Prerequisites

### Install Dependencies
```bash
cd src/rl-router
pip install -r requirements.txt
```

### Verify MongoDB Connection
Ensure MongoDB is running and accessible:
```bash
# Check MongoDB status
docker ps | grep mongo

# Or if running locally
mongosh --eval "db.adminCommand('ping')"
```

## Test Suite Overview

### 1. Unit Tests - Metrics Tracker
**File**: `test_metrics_tracker.py`

Tests the metrics tracking infrastructure:
- HopMetrics creation and serialization
- RouteMetrics aggregation
- MetricsComparator statistics

**Run:**
```bash
python -m unittest test_metrics_tracker.py -v
```

**Expected Output:**
```
test_add_hop (test_metrics_tracker.TestRouteMetrics) ... ok
test_add_metrics (test_metrics_tracker.TestMetricsComparator) ... ok
test_calculate_improvement (test_metrics_tracker.TestMetricsComparator) ... ok
...
----------------------------------------------------------------------
Ran 10 tests in 0.XXXs

OK
```

### 2. Environment Tests
**File**: `test_satellite_env.py`

Tests the satellite environment simulation:
- State reset
- Step function
- Reward calculation

**Run:**
```bash
python test_satellite_env.py
```

**Expected Output:**
```
INFO - Reset state vector: [array with 94 values]
INFO - Next state vector: [array with 94 values]
INFO - Reward: XX.XXX
INFO - Done: True/False
```

### 3. Comparison Tests - RL vs Dijkstra
**File**: `test_rl_vs_dijkstra.py`

Comprehensive comparison of routing algorithms:
- Multiple topology tests
- Performance metrics collection
- Statistical analysis

**Run:**
```bash
python test_rl_vs_dijkstra.py
```

**Expected Output:**
```
================================================================================
RL vs DIJKSTRA ROUTING COMPARISON TEST
================================================================================
Initializing components...
Found XX nodes in network
Generating 30 test packets...

Running RL routing tests...
  RL Test 1/30: NODE-A -> NODE-Z
  ...

Running Dijkstra routing tests...
  Dijkstra Test 1/30: NODE-A -> NODE-Z
  ...

================================================================================
ROUTING ALGORITHM COMPARISON SUMMARY
================================================================================

📊 RL-DQN Performance:
  Algorithm:              RL-DQN
  Total Packets:          30
  Successful Deliveries:  XX
  Delivery Rate:          XX.XX%
  Average Latency:        XX.XX ms
  ...

📊 Dijkstra Performance:
  Algorithm:              DIJKSTRA
  Total Packets:          30
  Successful Deliveries:  XX
  Delivery Rate:          XX.XX%
  Average Latency:        XX.XX ms
  ...

📈 Improvement Analysis:
  Latency Reduction:        +XX.XX%
  Hop Count Reduction:      +XX.XX%
  Resource Optimization:    +XX.XX%
  Delivery Rate Change:     +XX.XX%
================================================================================

Detailed comparison results saved to: comparison_results.json
Test completed successfully!
```

## Interpreting Results

### Success Criteria

The RL algorithm is considered successful if:

1. **Latency**: Average latency ≤ Dijkstra (target: 10-20% reduction)
2. **Delivery Rate**: Success rate ≥ Dijkstra (target: ≥99%)
3. **Resource Usage**: More balanced utilization (target: 15-25% improvement)
4. **Hop Count**: Similar or fewer hops (target: neutral to 15% reduction)

### Metrics Breakdown

#### Latency Metrics
- **Average Latency**: Mean end-to-end delay
- **Median Latency**: Middle value (less affected by outliers)
- **Std Latency**: Consistency indicator (lower is better)

#### Resource Metrics
- **Avg Node Utilization**: Average resource usage across hops
- **Max Node Utilization**: Peak usage (lower indicates better load balancing)

#### Reliability Metrics
- **Delivery Rate**: Percentage of successfully delivered packets
- **Drop Reasons**: Categorized failure modes

## Training the Optimized Model

Before running comparison tests, train the model with optimized parameters:

```bash
# Start training
python main_train.py

# Monitor training progress
# - Episode rewards should increase
# - Epsilon should decay from 0.95 to 0.01
# - Coverage should reach high percentage

# Training saves checkpoints to:
# models/checkpoints/dqn_checkpoint_fullpath_latest.pth
```

**Training Parameters:**
- Episodes: 20,000 (can adjust based on convergence)
- Batch Size: 128
- Buffer Size: 200,000
- Learning Rate: 3e-6
- Epsilon Decay: 100,000 steps

**Expected Training Time:**
- CPU: 4-8 hours
- GPU: 1-3 hours

## Automated Testing Workflow

### Full Test Suite
```bash
#!/bin/bash
# Run all tests in sequence

echo "1. Testing metrics tracker..."
python -m unittest test_metrics_tracker.py

echo "2. Testing environment simulation..."
python test_satellite_env.py

echo "3. Running RL vs Dijkstra comparison..."
python test_rl_vs_dijkstra.py

echo "All tests completed!"
```

Save as `run_all_tests.sh` and execute:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## Continuous Validation

### During Training
Monitor these metrics every 500 episodes:
1. Average reward trend (should increase)
2. Epsilon decay (should decrease)
3. Network topology coverage

### After Training
1. Run comparison tests on diverse packet pairs
2. Test on different network topologies
3. Validate QoS constraint compliance
4. Check resource balance across nodes

## Troubleshooting

### Issue: Import Errors
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: MongoDB Connection Failed
**Solution**: Check MongoDB is running
```bash
docker-compose up -d mongodb
# or
systemctl start mongod
```

### Issue: No Nodes Found in Database
**Solution**: Seed the database first
```bash
# Run seeding script (if available)
python seed_network_data.py
```

### Issue: RL Performance Worse than Dijkstra
**Possible Causes:**
1. Model not trained enough → Train for more episodes
2. Reward weights need tuning → Adjust in `satellite_simulator.py`
3. Network topology too sparse → Check connectivity
4. Epsilon still too high → Wait for more training steps

**Debug Steps:**
```python
# Check current epsilon
from python.rl_agent.policy import get_epsilon
epsilon = get_epsilon(steps_done)
print(f"Current epsilon: {epsilon}")  # Should be < 0.1 after training

# Verify model is loaded
import torch
checkpoint = torch.load('models/checkpoints/dqn_checkpoint_fullpath_latest.pth')
print(f"Trained episodes: {checkpoint['episode']}")
```

## Performance Benchmarks

### Expected Results (After Full Training)

| Metric | Dijkstra | RL-DQN | Improvement |
|--------|----------|--------|-------------|
| Avg Latency (ms) | 100-150 | 80-120 | 10-20% ↓ |
| Delivery Rate (%) | 95-98 | 96-99 | 1-2% ↑ |
| Avg Hops | 3-5 | 3-5 | Similar |
| Resource Balance | Uneven | Balanced | 15-25% ↑ |

### Validation Checklist

- [ ] All unit tests pass
- [ ] RL delivery rate ≥ Dijkstra
- [ ] RL average latency ≤ Dijkstra
- [ ] Resource utilization more balanced
- [ ] QoS violations reduced
- [ ] Results documented in `comparison_results.json`
- [ ] Performance meets target improvements

## Advanced Testing

### Stress Testing
Test with high packet load:
```python
# In test_rl_vs_dijkstra.py, modify:
num_test_packets = 100  # Increase from 30
```

### Multi-Topology Testing
Test across different network configurations:
- Dense networks (many neighbors)
- Sparse networks (few neighbors)
- Mixed altitude topologies (LEO/MEO/GEO)

### Adversarial Testing
Test under challenging conditions:
- High latency links
- Node failures
- Congestion scenarios
- Weather events

## Reporting

After testing, generate a report:
```bash
# Results are automatically saved
cat comparison_results.json | jq '.'

# Or use Python to generate formatted report
python -c "
import json
with open('comparison_results.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"
```

---

**Note**: All tests assume you have:
1. MongoDB running with seeded network data
2. Python 3.8+ with all dependencies installed
3. Sufficient system resources (8GB RAM minimum)

For production deployment, run the full test suite and ensure all criteria are met before replacing Dijkstra routing.
