# RL Optimization Summary

## Mục tiêu
Tối ưu RL để tìm được paths ngắn hơn, giảm distance so với Dijkstra baseline.

## Vấn đề ban đầu
- RL tìm được paths với distance lớn hơn Dijkstra rất nhiều (ví dụ: 73463.7km vs 18648.0km)
- Nhiều trường hợp "Max steps reached" - không tìm được path
- RL đôi khi ưu tiên quality/resource hơn distance

## Các tối ưu đã thực hiện

### 1. Tăng Progress Reward (khuyến khích tiến gần destination)
- `PROGRESS_REWARD_SCALE`: 80.0 → **120.0** (+50%)
- Khuyến khích agent di chuyển gần hơn đến destination

### 2. Tăng Distance Penalty (phạt paths dài)
- `DISTANCE_REWARD_SCALE`: 10.0 → **20.0** (+100%)
- `DISTANCE_PENALTY_DIVISOR_M`: 10000000.0 → **5000000.0** (penalty mạnh hơn x2)
- Mỗi hop distance sẽ bị penalty mạnh hơn

### 3. Tăng Detour Penalty (phạt đi lệch)
- `DETOUR_PENALTY_DIVISOR_M`: 50000.0 → **30000.0** (penalty mạnh hơn)
- `DETOUR_PENALTY_MULTIPLIER`: 5.0 → **8.0** (+60%)
- Phạt mạnh hơn khi agent đi xa khỏi destination

### 4. Giảm Quality Reward (ưu tiên distance hơn quality)
- `QUALITY_REWARD_SCALE`: 10.0 → **5.0** (-50%)
- Agent sẽ ưu tiên distance hơn quality của nodes

### 5. Thêm Distance-Based Reward trong mỗi step
- Thêm reward dựa trên tổng distance tích lũy so với direct distance
- Penalty nếu `distance_ratio > DISTANCE_RATIO_POOR` (2.5)
- Bonus nếu `distance_ratio < DISTANCE_RATIO_EFFICIENT` (1.2)
- Khuyến khích paths ngắn hơn ngay trong quá trình routing

### 6. Tăng Distance Efficiency Rewards
- `BONUS_DISTANCE_EFFICIENT`: 30.0 → **50.0** (+67%)
- `BONUS_DISTANCE_ACCEPTABLE`: 15.0 → **25.0** (+67%)
- `PENALTY_DISTANCE_POOR`: -20.0 → **-40.0** (penalty mạnh hơn x2)
- `DISTANCE_RATIO_POOR`: 3.0 → **2.5** (penalty sớm hơn)

### 7. Điều chỉnh Node Scoring Weights
- `SCORE_WEIGHT_DIST_TO_DEST`: 0.7 → **0.85** (+21%)
- `SCORE_WEIGHT_DIST_TO_CURRENT`: 0.1 → **0.15** (+50%)
- `SCORE_WEIGHT_QUALITY`: 500000.0 → **300000.0** (-40%)
- State builder sẽ ưu tiên distance hơn quality khi filter nodes

## Kết quả mong đợi

Sau khi tối ưu:
- ✅ RL sẽ tìm được paths với distance ngắn hơn
- ✅ RL sẽ ít bị "Max steps reached" hơn (do progress reward mạnh hơn)
- ✅ RL sẽ ưu tiên distance hơn quality/resource
- ✅ RL sẽ có performance gần với Dijkstra hơn về distance

## Files đã sửa

1. **`Backend/environment/constants.py`**:
   - Tăng progress và distance reward scales
   - Giảm distance penalty divisors
   - Tăng distance efficiency rewards
   - Điều chỉnh scoring weights

2. **`Backend/environment/routing_env.py`**:
   - Thêm distance-based reward trong mỗi step
   - Tính toán và reward dựa trên tổng distance tích lũy

## Lưu ý

- Các thay đổi này sẽ ảnh hưởng đến training, nên cần retrain model để thấy hiệu quả
- Có thể cần điều chỉnh thêm các tham số nếu kết quả chưa đạt mong đợi
- Nên test với notebook `013_test_end_to_end_routing.ipynb` để so sánh với Dijkstra

## Next Steps

1. Retrain RL model với các reward mới
2. Test và so sánh với Dijkstra
3. Điều chỉnh thêm nếu cần

