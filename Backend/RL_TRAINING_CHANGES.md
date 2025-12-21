# 🔧 RL Training Optimization - Changelog & Recommendations

## Overview

Tài liệu này mô tả các thay đổi đã thực hiện để cải thiện hiệu suất RL training và đề xuất hướng phát triển tiếp theo.

---

## 📝 Changelog

### 1. Config Changes (`config.dev.yaml`)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `learning_starts` | 1000 | **256** | Bắt đầu học sớm hơn |
| `target_update_interval` | 1000 | **100** | Q-targets ổn định hơn |
| `batch_size` | 64 | **32** | Updates nhanh hơn |
| `exploration_final_eps` | 0.01 | **0.05** | Exploration nhiều hơn |
| `epsilon_decay_strategy` | N/A | **"linear"** | Decay ổn định |
| `max_nodes` | 53 | **15** | State vector nhỏ hơn |
| `node_feature_dim` | 18 | **12** | Features tinh gọn |

**State dimension**: 994 → **200** (giảm ~5x)

---

### 2. Epsilon Decay (`dueling_dqn.py`)

**Before**: Exponential decay quá nhanh
```python
epsilon = epsilon_start * (0.9995 ** total_steps)
```

**After**: Linear decay ổn định
```python
if strategy == 'linear':
    progress = total_steps / max_steps
    epsilon = start - progress * (start - end)
```

---

### 3. Reward Function (`routing_env.py`)

**Before**: 15+ reward components → rewards -95000 đến +2000

**After**: 3 core components → rewards **-52 to +580**

| Case | Components |
|------|------------|
| **Per-step** | Progress ratio × 30, Step penalty -1, Util penalty -20 |
| **Success** | Base 500, Dest GS +50, Efficiency ±30 |
| **Truncated** | Fixed -50 |

**Key change**: Ratio-based rewards thay vì absolute distances:
```python
progress_ratio = progress / initial_distance  # -1 to +1
reward = progress_ratio * 30.0
```

---

## 🎯 Recommendations

### Short-term (1-2 days)

1. **Run longer training**: 5000+ episodes với config mới
2. **Monitor metrics**:
   - Loss nên < 1000
   - Success rate nên > 70%
   - Avg reward nên > 0

### Medium-term (1 week)

1. **Add directional features** to state:
   ```python
   delta_lat = (dest_lat - node_lat) / 180.0
   delta_lon = (dest_lon - node_lon) / 360.0
   ```

2. **Implement curriculum learning**:
   - Start với easy pairs (gần nhau)
   - Gradually tăng difficulty

3. **Tune reward scales**:
   - Progress: 30 → 50 nếu cần stronger signal
   - Step penalty: -1 → -2 nếu paths quá dài

### Long-term (2-4 weeks)

1. **Imitation learning warmup**:
   - Pre-train với Dijkstra expert paths
   - Sau đó fine-tune với RL

2. **Multi-objective optimization**:
   - Separate heads cho distance vs utilization
   - Weighted combination

3. **Graph Neural Network**:
   - Thay MLP bằng GNN cho tốt hơn với graph structure
   - Xem xét GAT (Graph Attention) hoặc GCN

---

## 📊 Expected Results

| Metric | Before | Expected After |
|--------|--------|----------------|
| Success Rate | 40% | **>70%** |
| Loss | 18000+ | **<1000** |
| Avg Reward | -50000 | **>0** |
| Training Speed | Slow | **~3x faster** |

---

## 🔗 Related Files

- [config.dev.yaml](./config.dev.yaml) - Training configuration
- [dueling_dqn.py](./agent/dueling_dqn.py) - Agent implementation
- [routing_env.py](./environment/routing_env.py) - Environment & rewards
- [state_builder.py](./environment/state_builder.py) - State representation
