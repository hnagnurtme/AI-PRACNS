# Packet Routing - Hướng dẫn sử dụng

## 📦 Tính năng Packet Routing

Tính năng **Packet Routing** cho phép bạn:
1. Chọn **Source Terminal** (nguồn) và **Destination Terminal** (đích)
2. Tính toán đường đi (path) từ nguồn đến đích qua các nodes
3. Gửi packet và xem path được vẽ trên map

## 🎯 Cách sử dụng

### Bước 1: Mở Packet Sender
- Click nút **"📦 Send Packet"** ở góc trên bên trái của map
- Panel **Packet Sender** sẽ hiện ra

### Bước 2: Chọn Source Terminal
- Trong dropdown **"Source Terminal"**, chọn terminal nguồn
- Terminal này sẽ là điểm bắt đầu của packet

### Bước 3: Chọn Destination Terminal
- Trong dropdown **"Destination Terminal"**, chọn terminal đích
- Terminal này sẽ là điểm kết thúc của packet

### Bước 4: Cấu hình Packet (tùy chọn)
- **Packet Size**: Kích thước packet (bytes), mặc định: 1024
- **Priority**: Độ ưu tiên (1-10), mặc định: 5

### Bước 5: Tính toán Path hoặc Gửi Packet

#### Option 1: Calculate Path
- Click **"Calculate Path"** để xem đường đi mà không gửi packet
- Path sẽ được vẽ trên map với:
  - **Đường màu cyan**: Đường đi từ source đến destination
  - **Marker xanh lá (SOURCE)**: Điểm nguồn
  - **Marker đỏ (DEST)**: Điểm đích
  - **Marker cyan**: Các nodes trung gian

#### Option 2: Send Packet
- Click **"Send Packet"** để gửi packet và tự động tính toán path
- Packet sẽ được gửi đến backend
- Path sẽ được vẽ trên map
- Thông tin packet sẽ hiển thị trong panel

## 📊 Thông tin hiển thị

### Path Information
Sau khi tính toán path, bạn sẽ thấy:
- **Hops**: Số lượng bước nhảy (terminal → node → node → terminal)
- **Distance**: Tổng khoảng cách (km)
- **Estimated Latency**: Độ trễ ước tính (ms)

### Packet Information
Sau khi gửi packet:
- **Packet ID**: ID duy nhất của packet
- **Status**: Trạng thái (sent, in_transit, delivered, failed)
- **ETA**: Thời gian ước tính đến đích

## 🗺️ Visualization trên Map

Path được vẽ trên Cesium map với:

1. **Polyline màu cyan**: Đường đi từ source đến destination
   - Đi qua các nodes trung gian
   - Hiển thị toàn bộ route

2. **Markers**:
   - 🟢 **SOURCE** (xanh lá): Terminal nguồn
   - 🔴 **DEST** (đỏ): Terminal đích
   - 🔵 **Nodes** (cyan): Các nodes trung gian với tên

3. **Labels**: Tên của các nodes trung gian

## 🔧 API Endpoints

### Calculate Path
```http
POST /api/v1/routing/calculate-path
Content-Type: application/json

{
  "sourceTerminalId": "TERM-xxx",
  "destinationTerminalId": "TERM-yyy"
}
```

**Response:**
```json
{
  "source": {
    "terminalId": "TERM-xxx",
    "position": { "latitude": ..., "longitude": ..., "altitude": ... }
  },
  "destination": {
    "terminalId": "TERM-yyy",
    "position": { ... }
  },
  "path": [
    { "type": "terminal", "id": "...", "name": "...", "position": {...} },
    { "type": "node", "id": "...", "name": "...", "position": {...} },
    ...
  ],
  "totalDistance": 1234.56,
  "estimatedLatency": 89.12,
  "hops": 4
}
```

### Send Packet
```http
POST /api/v1/routing/send-packet
Content-Type: application/json

{
  "sourceTerminalId": "TERM-xxx",
  "destinationTerminalId": "TERM-yyy",
  "packetSize": 1024,
  "priority": 5
}
```

**Response:**
```json
{
  "packetId": "PKT-1234567890",
  "sourceTerminalId": "TERM-xxx",
  "destinationTerminalId": "TERM-yyy",
  "packetSize": 1024,
  "priority": 5,
  "path": { ... },
  "status": "sent",
  "sentAt": "2025-11-26T...",
  "estimatedArrival": "2025-11-26T..."
}
```

## 🧮 Thuật toán Routing

Backend sử dụng thuật toán routing đơn giản:

1. **Tìm node gần nhất** cho source terminal
2. **Tìm node gần nhất** cho destination terminal
3. **Kiểm tra khoảng cách**:
   - Nếu khoảng cách < 2x maxRange: Kết nối trực tiếp
   - Nếu khoảng cách > 2x maxRange: Tìm node trung gian
4. **Tính toán path** qua các nodes
5. **Tính toán metrics**:
   - Total distance (km)
   - Estimated latency (ms) = propagation delay + processing delay
   - Number of hops

## 💡 Use Cases

1. **Network Testing**: Test khả năng routing của mạng SAGIN
2. **Path Analysis**: Phân tích đường đi tối ưu giữa các terminals
3. **QoS Evaluation**: Đánh giá chất lượng dịch vụ (latency, distance)
4. **Network Planning**: Lập kế hoạch mạng với nhiều terminals

## 🔄 Tương lai

Có thể mở rộng với:
- Thuật toán routing phức tạp hơn (Dijkstra, A*, QoS-based)
- Real-time packet tracking
- Multiple paths comparison
- Load balancing
- Congestion avoidance

## 📝 Notes

- Path được tính toán dựa trên vị trí hiện tại của nodes
- Với satellites đang di chuyển, path có thể thay đổi theo thời gian
- Latency là ước tính dựa trên khoảng cách và processing delay
- Packet size và priority hiện tại chưa ảnh hưởng đến routing (có thể mở rộng)

