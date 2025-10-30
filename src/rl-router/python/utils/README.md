# RL Router Utils

This folder contains **MongoDB connector** and **StateBuilder** modules for RL network simulations.

---

## 1. `db_connector.py`

Module này chịu trách nhiệm **kết nối và truy vấn dữ liệu từ MongoDB**.

### Class `MongoConnector`

Quản lý kết nối và truy vấn Node trong MongoDB.

#### Các hàm chính:

| Function | Mô tả |
|----------|-------|
| `get_node(node_id, projection=None)` | Lấy 1 node theo `nodeId`. |
| `get_all_nodes(projection=None)` | Lấy tất cả node trong collection. |
| `get_nodes_by_type(node_type, projection=None)` | Lấy tất cả node theo `nodeType`. |
| `get_nodes_by_status(operational, projection=None)` | Lấy tất cả node theo trạng thái hoạt động. |
| `get_neighbor_status_batch(neighbor_ids, projection=None)` | Lấy chi tiết trạng thái tất cả neighbors theo danh sách ID (batch, tối ưu cho RL). |
| `get_node_neighbors(node_id, projection=None)` | Lấy tất cả neighbor của một node dưới dạng dict (nodeId → NodeData). |

**Note:** `projection` là tùy chọn để chọn field cần fetch, ví dụ `{"position": 1}`.

---

## 2. `state_builder.py`

Module này **xây dựng và chuẩn hóa Vector Trạng thái (S)** dùng trực tiếp cho RL agent.

### Class `StateBuilder`

- Nhận dữ liệu packet và node từ MongoConnector.
- Trả về vector numpy float32 chuẩn hóa với kích thước cố định.
- Vector gồm 4 thành phần chính:

| Component | Fields | Size | Notes |
|-----------|--------|------|-------|
| **V_G: QoS + Packet progress** | One-hot service type | 5 | VIDEO_STREAM, AUDIO_CALL,... |
|  | max latency / MAX_SYSTEM_LATENCY_MS | 1 | Normalized |
|  | min bandwidth / MAX_LINK_BANDWIDTH_MBPS | 1 | Normalized |
|  | max loss rate | 1 | Raw 0-1 |
|  | accumulated delay / max latency | 1 | Progress ratio |
|  | TTL / max TTL | 1 | Normalized |
| **V_P: Node position + direction** | One-hot node type | 4 | GROUND_STATION, LEO, MEO, GEO |
|  | distance to destination / MAX_SYSTEM_DISTANCE_KM | 1 | Normalized |
|  | direction vector (x,y,z) | 3 | Unit vector |
| **V_C: Current node state** | Resource utilization | 1 | 0-1 |
|  | Buffer ratio | 1 | currentPacketCount / capacity |
|  | Packet loss rate | 1 | 0-1 |
|  | Node processing delay / MAX_PROCESSING_DELAY_MS | 1 | Normalized |
|  | Data staleness | 1 | 0-1 |
|  | Operational status | 1 | 1=active, 0=down |
| **V_N: Neighbors (MAX_NEIGHBORS=4)** | For each neighbor: operational, total latency / MAX, available bandwidth / MAX, utilization, packet loss, distance to dest / MAX, FSPL / 300 | 7 × 4 = 28 | Padding 0 nếu neighbor thiếu |

- **Vector tổng cộng:** 52 phần tử.

---

## 3. How to Test
```
1. Tạo **mock data**: node hiện tại, node đích, neighbors, packet.
2. Khởi tạo `MongoConnector` giả lập hoặc dùng Mock object.
3. Tạo `StateBuilder` và gọi `get_state_vector(packet_data)`.
4. Kiểm tra:
   - Kích thước vector = 52.
   - Giá trị chuẩn hóa trong `[0,1]`.
   - One-hot encoding đúng.
   - Neighbor padding đúng.

```
**Chạy test**
```
python main_test_state_builder.py
python main_test.py
```