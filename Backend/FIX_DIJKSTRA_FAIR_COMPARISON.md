# Fix: Dijkstra Fair Comparison với RL

## Vấn đề

Trong test notebook `013_test_end_to_end_routing.ipynb`, có một lỗi logic nghiêm trọng khiến so sánh giữa RL và Dijkstra không công bằng:

1. **Test notebook** chọn Ground Stations bằng `find_best_ground_station()` cho cả RL và Dijkstra
2. **RL** sử dụng các GS đã chọn này một cách rõ ràng
3. **Dijkstra** (`calculate_path_dijkstra`) **bỏ qua** các GS đã chọn và tự chọn GS mới bằng `find_nearest_ground_station()`

**Hệ quả**: RL và Dijkstra đang giải quyết **2 bài toán routing khác nhau** (từ GS khác nhau), nên không thể so sánh công bằng!

## Nguyên nhân

- `calculate_path_dijkstra()` luôn tự chọn GS bằng `find_nearest_ground_station()` bên trong hàm
- Test notebook không thể truyền GS đã chọn vào Dijkstra
- Điều này dẫn đến việc RL có thể tìm được path ngắn hơn Dijkstra (vì bắt đầu từ GS khác nhau), điều này **KHÔNG HỢP LÝ** vì Dijkstra phải luôn tìm được path ngắn nhất (từ GS đã chọn)

## Giải pháp

### 1. Sửa `calculate_path_dijkstra()` trong `Backend/api/routing_bp.py`

Thêm 2 tham số tùy chọn `source_gs` và `dest_gs` để cho phép truyền GS đã chọn sẵn:

```python
def calculate_path_dijkstra(source_terminal: dict, dest_terminal: dict, nodes: list, 
                           resource_aware: bool = False, drop_threshold: float = 95.0,
                           penalty_threshold: float = 80.0, penalty_multiplier: float = 3.0,
                           source_gs: Optional[dict] = None, dest_gs: Optional[dict] = None) -> dict:
```

Logic:
- Nếu `source_gs` và `dest_gs` được cung cấp: Dùng GS đó (để so sánh công bằng với RL)
- Nếu không: Dùng `find_nearest_ground_station()` như cũ (baseline)

### 2. Sửa test notebook `013_test_end_to_end_routing.ipynb`

Cập nhật cell 11 (test function) để truyền cùng GS cho Dijkstra:

```python
dijkstra_path = calculate_path_dijkstra(
    source_terminal,
    dest_terminal,
    nodes,
    resource_aware=True,
    source_gs=source_gs,  # Use same GS as RL
    dest_gs=dest_gs        # Use same GS as RL
)
```

## Kết quả mong đợi

Sau khi sửa:
- ✅ RL và Dijkstra giải quyết **CÙNG MỘT bài toán routing** (từ cùng GS nguồn đến cùng GS đích)
- ✅ Dijkstra **PHẢI** luôn tìm được path với distance ngắn nhất (vì tối ưu pure distance)
- ✅ Nếu RL tìm được path ngắn hơn Dijkstra, đó là **LỖI LOGIC** cần được điều tra thêm
- ✅ So sánh giữa RL và Dijkstra giờ đây là **CÔNG BẰNG** và có ý nghĩa

## Lưu ý

- Nếu sau khi sửa, Dijkstra vẫn tìm được path dài hơn RL, cần kiểm tra:
  1. Logic xây dựng graph của Dijkstra có đúng không?
  2. Edge weights có đúng là pure distance không?
  3. RL có đang terminate sớm và skip satellite routing không?
  4. Cách tính distance của RL và Dijkstra có giống nhau không?

## Files đã sửa

1. `Backend/api/routing_bp.py`: Thêm tham số `source_gs` và `dest_gs` cho `calculate_path_dijkstra()`
2. `Backend/notebooks/013_test_end_to_end_routing.ipynb`: Cập nhật test function để truyền cùng GS cho Dijkstra

