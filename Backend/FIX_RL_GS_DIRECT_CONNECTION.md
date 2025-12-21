# Fix: RL Logic Error - Ground Station Direct Connection

## Vấn đề

RL đang tìm được path không hợp lý: **Ground Station kết nối trực tiếp với Ground Station khác ở khoảng cách xa** mà không đi qua satellites.

**Ví dụ lỗi**:
```
RL Path: GS-033 (New York) → GS-035 (Chicago) - 1715.93km
Dijkstra Path: GS-033 → MEO-026 → GS-035 - 17401.34km (đúng)
```

**Vấn đề**: New York không thể gửi trực tiếp đến Chicago (1715km) mà phải đi qua satellites.

## Nguyên nhân

Trong `Backend/environment/state_builder.py`, hàm `_smart_node_filtering()` có logic sai:

```python
# Logic CŨ (SAI):
if current_node_type == 'GROUND_STATION' and node_type == 'GROUND_STATION':
    if dist_to_current > max_range * GS_RANGE_MARGIN:  # max_range = 2000km
        return float('inf')
```

Logic này cho phép Ground Stations kết nối trực tiếp nếu distance <= 2000km, điều này **KHÔNG HỢP LÝ** vì:
- Ground Stations chỉ có thể kết nối trực tiếp nếu rất gần (< 100km)
- Nếu distance > 100km, **PHẢI** đi qua satellites

## Giải pháp

Sửa logic trong `_smart_node_filtering()` để:
1. Ground Stations chỉ có thể kết nối trực tiếp nếu distance < 100km (`GS_DIRECT_CONNECTION_THRESHOLD_KM`)
2. Nếu distance > 100km, Ground Stations **PHẢI** đi qua satellites (giống Dijkstra)

```python
# Logic MỚI (ĐÚNG):
if current_node_type == 'GROUND_STATION' and node_type == 'GROUND_STATION':
    dist_to_current_km = dist_to_current / M_TO_KM
    if dist_to_current_km > GS_DIRECT_CONNECTION_THRESHOLD_KM:  # 100km
        # Không cho phép direct GS-to-GS connection nếu distance > 100km
        # Phải đi qua satellites
        return float('inf')
```

## Files đã sửa

1. **`Backend/environment/state_builder.py`**:
   - Sửa logic trong `_smart_node_filtering()` để chỉ cho phép GS kết nối trực tiếp nếu distance < 100km
   - Thêm import `GS_DIRECT_CONNECTION_THRESHOLD_KM` từ `environment.constants`

## Kết quả mong đợi

Sau khi sửa:
- ✅ RL **KHÔNG THỂ** tìm được path với direct GS-to-GS connection nếu distance > 100km
- ✅ RL **PHẢI** đi qua satellites giống Dijkstra
- ✅ RL và Dijkstra sẽ có path tương tự nhau (cùng đi qua satellites)
- ✅ So sánh giữa RL và Dijkstra giờ đây là **CÔNG BẰNG** và **HỢP LÝ**

## Lưu ý

- Logic này giống với logic của Dijkstra trong `calculate_path_dijkstra()`:
  ```python
  if node_type == 'GROUND_STATION' and other_type == 'GROUND_STATION':
      if distance_km <= GS_DIRECT_CONNECTION_THRESHOLD_KM:  # 100km
          # Allow direct connection
      else:
          # Must go through satellites
          continue
  ```

- Đảm bảo tính nhất quán giữa RL và Dijkstra về communication constraints.

