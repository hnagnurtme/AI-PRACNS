# Report: Routing Logic Issues - RL vs Dijkstra

## 🚨 Vấn Đề Chính

**Dijkstra PHẢI luôn tìm được path với ít hops nhất** (hoặc distance ngắn nhất tùy vào edge weights), nhưng hiện tại có thể không đúng do:

1. **Objective Mismatch**: Dijkstra tối ưu weighted distance (distance × resource_factor), không phải số hops
2. **RL có thể tìm được path với ít hops hơn**: Do reward function khuyến khích tối ưu hops
3. **Resource Penalties**: Làm edge weights lớn, khiến Dijkstra chọn path dài hơn để tránh nodes có resource cao

---

## 📊 Phân Tích Chi Tiết

### 1. Dijkstra Algorithm Logic

#### Current Implementation
```python
# Backend/api/routing_bp.py - calculate_path_dijkstra()

# Edge weights = distance_km * resource_factor
def calculate_edge_weight(node, other_node, base_distance_km):
    if not resource_aware:
        return base_distance_km  # Pure distance
    
    util = get_node_utilization(other_node)
    
    # Resource penalties:
    # - Low utilization (<40%): factor = 0.95 (bonus)
    # - Medium (40-60%): factor = 1.0-1.2
    # - High (60-80%): factor = 1.0-1.5
    # - Very high (>=80%): factor = 1.0-3.0x
    
    resource_factor = calculate_resource_factor(util)
    return base_distance_km * resource_factor
```

#### Vấn Đề
- **Dijkstra tối ưu**: Weighted distance (distance × resource_factor)
- **KHÔNG tối ưu**: Số hops
- **Kết quả**: Có thể chọn path dài hơn (nhiều hops) nếu weighted distance nhỏ hơn

#### Ví Dụ
```
Path A: GS1 → Sat1 → GS2 (3 hops, distance=5000km, util=90% → weight=5000×3.0=15000)
Path B: GS1 → Sat1 → Sat2 → Sat3 → GS2 (5 hops, distance=6000km, util=30% → weight=6000×0.95=5700)

Dijkstra chọn Path B (weighted distance nhỏ hơn) dù có nhiều hops hơn!
```

---

### 2. RL Algorithm Logic

#### Current Implementation
```python
# Backend/environment/routing_env.py - step()

# Reward function:
reward = REWARD_SUCCESS  # 500.0

# Efficiency rewards (khuyến khích ít hops):
if num_hops <= optimal_hops:
    efficiency_bonus = (optimal_hops - num_hops) * EFFICIENCY_BONUS_PER_HOP  # +20 per hop saved
    reward += efficiency_bonus
else:
    efficiency_penalty = (num_hops - optimal_hops) * EFFICIENCY_PENALTY_PER_HOP  # -15 per extra hop
    reward -= efficiency_penalty

# Distance efficiency:
distance_ratio = total_distance / direct_distance
if distance_ratio <= DISTANCE_RATIO_EFFICIENT:  # <= 1.2
    reward += BONUS_DISTANCE_EFFICIENT  # +30
elif distance_ratio <= DISTANCE_RATIO_ACCEPTABLE:  # <= 1.5
    reward += BONUS_DISTANCE_ACCEPTABLE  # +15
else:
    reward += PENALTY_DISTANCE_POOR  # -20
```

#### Vấn Đề
- **RL tối ưu**: Multi-objective (hops, distance, resource quality, QoS)
- **Reward khuyến khích**: Ít hops (EFFICIENCY_BONUS_PER_HOP = +20)
- **Kết quả**: RL có thể tìm được path với ít hops hơn Dijkstra

---

## 🔍 Root Cause Analysis

### Vấn Đề 1: Objective Mismatch

| Algorithm | Objective | Metric |
|-----------|-----------|--------|
| **Dijkstra** | Minimize weighted distance | `distance × resource_factor` |
| **RL** | Maximize reward (multi-objective) | `hops, distance, resource, QoS` |

**Kết quả**: Hai algorithms tối ưu các metrics khác nhau → không thể so sánh công bằng!

### Vấn Đề 2: Resource Penalties Quá Lớn

```python
# Resource penalties trong Dijkstra:
if util >= 80%:
    resource_factor = 1.0 + (util - 80) / 20 * 2.0  # Up to 3.0x
```

**Vấn đề**: Penalty 3.0x có thể làm edge weight lớn hơn nhiều so với distance thực tế, khiến Dijkstra chọn path dài hơn để tránh nodes có resource cao.

### Vấn Đề 3: RL Reward Function Khuyến Khích Hops

```python
# RL reward khuyến khích ít hops:
EFFICIENCY_BONUS_PER_HOP = 20.0  # +20 per hop saved
EFFICIENCY_PENALTY_PER_HOP = 15.0  # -15 per extra hop
```

**Vấn đề**: RL được train để tối ưu số hops, trong khi Dijkstra tối ưu weighted distance → RL có thể tìm được path với ít hops hơn.

---

## 📈 Kết Quả Thực Tế

### Scenario: Terminal A → Terminal B

**Dijkstra (Baseline)**:
- Edge weights = distance × resource_factor
- Chọn path với weighted distance nhỏ nhất
- Có thể có nhiều hops nếu nodes có resource tốt (low penalty)

**RL (Optimized)**:
- Reward khuyến khích ít hops
- Chọn path với ít hops nhất (nếu có thể)
- Có thể có ít hops hơn Dijkstra

**Kết luận**: RL có thể tìm được path với **ít hops hơn** Dijkstra, điều này **VÔ LÝ** về mặt lý thuyết!

---

## ✅ Giải Pháp Đề Xuất

### Solution 1: Dijkstra Tối Ưu Hops (Unweighted Graph)

**Thay đổi**: Dijkstra dùng unweighted graph (edge weight = 1) để tối ưu số hops:

```python
def calculate_path_dijkstra_unweighted(...):
    # Edge weight = 1 (unweighted)
    graph[node_id].append((neighbor_id, 1.0))
    
    # Dijkstra sẽ tìm path với ít hops nhất
```

**Ưu điểm**:
- Dijkstra đảm bảo tìm được path với ít hops nhất
- Fair comparison với RL về số hops

**Nhược điểm**:
- Không tối ưu resource utilization
- Không phản ánh thực tế (distance và resource quan trọng)

### Solution 2: Dijkstra Tối Ưu Distance (Pure Distance)

**Thay đổi**: Dijkstra chỉ tối ưu distance, không có resource penalties:

```python
def calculate_path_dijkstra_pure_distance(...):
    # Edge weight = distance only (no resource factor)
    edge_weight = distance_km
    
    # Dijkstra sẽ tìm path với distance ngắn nhất
```

**Ưu điểm**:
- Dijkstra đảm bảo tìm được path với distance ngắn nhất
- Baseline rõ ràng (pure distance optimization)

**Nhược điểm**:
- Không tối ưu resource
- Có thể chọn nodes overloaded

### Solution 3: RL Tối Ưu Weighted Distance (Giống Dijkstra)

**Thay đổi**: RL reward function khuyến khích weighted distance thay vì hops:

```python
# RL reward = -weighted_distance (minimize)
weighted_distance = total_distance * avg_resource_factor
reward = -weighted_distance * DISTANCE_REWARD_SCALE
```

**Ưu điểm**:
- RL và Dijkstra cùng objective (weighted distance)
- Fair comparison

**Nhược điểm**:
- RL không tối ưu hops nữa
- Mất đi lợi ích của RL (multi-objective optimization)

### Solution 4: Hybrid Approach (Recommended)

**Thay đổi**: 
1. **Dijkstra Baseline**: Tối ưu distance (pure, no resource penalties)
2. **RL Optimized**: Tối ưu multi-objective (hops, distance, resource, QoS)

**So sánh**:
- Dijkstra: Path với distance ngắn nhất (baseline)
- RL: Path tối ưu multi-objective (có thể xa hơn nhưng resource tốt hơn, ít hops hơn)

**Ưu điểm**:
- Dijkstra đảm bảo distance ngắn nhất (baseline rõ ràng)
- RL thể hiện lợi ích của multi-objective optimization
- Fair comparison về distance (Dijkstra tốt hơn), nhưng RL tốt hơn về resource và hops

---

## 🎯 Recommendation

### Option A: Dijkstra Pure Distance (Baseline)

```python
def calculate_path_dijkstra(source_terminal, dest_terminal, nodes, 
                           resource_aware: bool = False):  # Default: False
    # Edge weight = distance only (no resource penalties)
    edge_weight = distance_km
    
    # Dijkstra tìm path với distance ngắn nhất
    # Đảm bảo: Dijkstra LUÔN tìm được path với distance ngắn nhất
```

**Kết quả**:
- Dijkstra: Path với distance ngắn nhất (baseline)
- RL: Path tối ưu multi-objective (có thể xa hơn nhưng resource tốt hơn)

### Option B: Dijkstra Unweighted (Hops Optimization)

```python
def calculate_path_dijkstra_unweighted(source_terminal, dest_terminal, nodes):
    # Edge weight = 1 (unweighted)
    edge_weight = 1.0
    
    # Dijkstra tìm path với ít hops nhất
    # Đảm bảo: Dijkstra LUÔN tìm được path với ít hops nhất
```

**Kết quả**:
- Dijkstra: Path với ít hops nhất (baseline)
- RL: Path tối ưu multi-objective (có thể nhiều hops hơn nhưng resource tốt hơn)

---

## 📝 Kết Luận

### Vấn Đề Hiện Tại
1. ❌ Dijkstra KHÔNG đảm bảo tìm được path với ít hops nhất
2. ❌ Dijkstra tối ưu weighted distance (distance × resource_factor)
3. ❌ RL có thể tìm được path với ít hops hơn Dijkstra (vô lý về mặt lý thuyết)
4. ❌ Không có fair comparison giữa RL và Dijkstra

### Giải Pháp
1. ✅ **Dijkstra Baseline**: Tối ưu distance (pure, no resource penalties)
2. ✅ **RL Optimized**: Tối ưu multi-objective (hops, distance, resource, QoS)
3. ✅ **Fair Comparison**: 
   - Dijkstra tốt hơn về distance (baseline)
   - RL tốt hơn về resource utilization và có thể ít hops hơn (optimization)

### Next Steps
1. Sửa `calculate_path_dijkstra()` để tối ưu pure distance (no resource penalties)
2. Hoặc tạo `calculate_path_dijkstra_unweighted()` để tối ưu hops
3. Update documentation và tests để phản ánh đúng logic

---

## 📚 References

- **Dijkstra's Algorithm**: Tìm đường đi ngắn nhất trong weighted graph
- **RL Multi-Objective Optimization**: Tối ưu nhiều objectives cùng lúc (hops, distance, resource, QoS)
- **Fair Comparison**: So sánh công bằng cần cùng objective hoặc rõ ràng về sự khác biệt

