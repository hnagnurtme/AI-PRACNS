# Routing Logic Comparison: RL vs Dijkstra

## Tổng Quan

Hệ thống routing sử dụng 2 algorithms chính để so sánh performance:

1. **RL (Reinforcement Learning)**: Tối ưu resource-aware routing
2. **Dijkstra (Baseline)**: Baseline algorithm chỉ xét khoảng cách

## Sự Khác Biệt Chính

### 1. Ground Station Selection

#### RL (Optimized) 🤖
- **Function**: `find_best_ground_station()`
- **Logic**: Tối ưu đa tiêu chí (multi-criteria optimization)
  - Distance: 25% weight (gần hơn = tốt hơn)
  - Resource Utilization: 25% weight (thấp hơn = tốt hơn)
  - Connection Count: 15% weight (ít hơn = load balancing tốt hơn)
  - Battery Level: 15% weight (cao hơn = tốt hơn)
  - Packet Loss Rate: 20% weight (thấp hơn = chất lượng tốt hơn)
- **Mục đích**: Chọn GS tốt nhất về resource, không chỉ khoảng cách
- **Kết quả**: RL có thể chọn GS xa hơn nhưng có resource tốt hơn

#### Dijkstra (Baseline) 📐
- **Function**: `find_nearest_ground_station()`
- **Logic**: Chỉ xét khoảng cách (distance-only)
  - Tìm GS gần nhất với terminal
  - Không quan tâm đến utilization, battery, packet loss, connections
- **Mục đích**: Baseline để so sánh với RL
- **Kết quả**: Luôn chọn GS gần nhất, có thể bị overload

### 2. Routing Algorithm

#### RL
- **Method**: DuelingDQN agent
- **State**: Multi-dimensional state vector (node features, network topology, QoS)
- **Action**: Select next node in path
- **Reward**: Multi-objective (distance, latency, resource utilization, QoS compliance)
- **Training**: Trained với curriculum learning và imitation learning

#### Dijkstra
- **Method**: Dijkstra's shortest path algorithm
- **Edge Weights**: Distance × Resource Factor (nếu `resource_aware=True`)
- **Resource Awareness**: 
  - Low utilization (<40%): 5% bonus (factor = 0.95)
  - Medium utilization (40-60%): slight penalty (factor = 1.0-1.2)
  - High utilization (60-80%): linear penalty (factor = 1.0-1.5)
  - Very high utilization (>=80%): exponential penalty (factor = 1.0-3.0x)
- **Node Dropping**: Nodes với utilization > 95% bị loại khỏi routing

## Code Implementation

### RL Routing
```python
# Backend/services/rl_routing_service.py
from api.routing_bp import find_best_ground_station

source_gs = find_best_ground_station(source_terminal, nodes)  # ✅ Tối ưu resource
dest_gs = find_best_ground_station(dest_terminal, nodes)       # ✅ Tối ưu resource
```

### Dijkstra Routing
```python
# Backend/api/routing_bp.py
def calculate_path_dijkstra(...):
    # 🔥 BASELINE: LUÔN dùng nearest GS (chỉ khoảng cách)
    source_node = find_nearest_ground_station(source_terminal, nodes)  # ✅ Chỉ khoảng cách
    dest_node = find_nearest_ground_station(dest_terminal, nodes)       # ✅ Chỉ khoảng cách
```

## Logging

### RL Logs
```
🤖 RL (OPTIMIZED): Selected BEST Ground Station GS-041 for terminal TERM-0008 
   (distance: 17.4km, utilization: 36.0%, battery: 100.0%, WITH resource optimization)
```

### Dijkstra Logs
```
📐 Dijkstra (BASELINE): Selected NEAREST Ground Station GS-042 for terminal TERM-0008 
   (distance: 15.2km, NO resource optimization)
```

## Kỳ Vọng Kết Quả

### Khi So Sánh RL vs Dijkstra:

1. **Success Rate**: RL có thể cao hơn vì tránh overloaded nodes
2. **Resource Utilization**: RL phân bổ load tốt hơn (load balancing)
3. **Latency**: Có thể tương đương hoặc tốt hơn (tùy network state)
4. **Hops**: RL có thể nhiều hơn một chút (để tránh congested paths)
5. **Reliability**: RL tốt hơn vì tránh nodes có vấn đề (low battery, high loss)

### Ví Dụ Thực Tế

**Scenario**: Terminal ở Hà Nội cần routing đến Terminal ở Hồ Chí Minh

**Dijkstra (Baseline)**:
- Chọn GS gần nhất ở Hà Nội (có thể đang overload 90%)
- Chọn GS gần nhất ở HCM (có thể đang overload 85%)
- Path có thể bị chậm do congestion

**RL (Optimized)**:
- Chọn GS tốt nhất ở Hà Nội (có thể xa hơn 5km nhưng utilization chỉ 30%)
- Chọn GS tốt nhất ở HCM (có thể xa hơn 3km nhưng utilization chỉ 25%)
- Path nhanh hơn và ổn định hơn do resource tốt

## Testing

Để test sự khác biệt, chạy:
```bash
# Test end-to-end routing
cd Backend/notebooks
jupyter notebook 013_test_end_to_end_routing.ipynb
```

Notebook này sẽ:
1. Test 50 cặp terminal ngẫu nhiên
2. So sánh RL vs Dijkstra
3. Phân tích sự khác biệt về:
   - GS selection (distance, utilization, battery)
   - Path metrics (hops, distance, latency)
   - Success rate
   - Resource utilization

## Kết Luận

- **RL**: Tối ưu resource-aware, chọn GS tốt nhất (multi-criteria)
- **Dijkstra**: Baseline, chọn GS gần nhất (distance-only)

Sự khác biệt này cho phép đánh giá được lợi ích của RL trong việc tối ưu resource utilization và load balancing so với baseline đơn giản.

