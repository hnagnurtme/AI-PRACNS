# Hệ Thống Truyền Thông Mạng SAGIN

Một nền tảng mô phỏng mạng tích hợp Không gian-Không khí-Mặt đất (SAGIN) phân tán cho phép truyền thông liền mạch qua chòm sao vệ tinh, trạm mặt đất và hạ tầng định tuyến lõi với tối ưu hóa bằng AI.

## 🚀 Tổng Quan Dự Án

Dự án này triển khai kiến trúc mạng SAGIN toàn diện với nhiều dịch vụ kết nối với nhau mô phỏng các tình huống truyền thông vệ tinh thực tế. Hệ thống cung cấp quản lý topology động, tối ưu hóa định tuyến thông minh và khả năng giám sát thời gian thực.

### Tính Năng Chính

- **Kiến trúc mạng đa tầng**: Dịch vụ vệ tinh, trạm mặt đất và bộ định tuyến lõi
- **Tối ưu hóa bằng AI**: Định tuyến và phân bổ tài nguyên dựa trên học tăng cường
- **Giám sát thời gian thực**: Tích hợp Firebase để theo dõi trạng thái mạng trực tiếp
- **Client tương tác**: Giao diện người dùng JavaFX để hiển thị mạng và chat
- **Thiết kế Cloud-Native**: Hỗ trợ triển khai Docker và Kubernetes đầy đủ
- **Tích hợp thời tiết**: Điều chỉnh topology động dựa trên điều kiện môi trường

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client GUI    │    │  Manager API    │    │  AI Optimizer   │
│    (JavaFX)     │◄──►│ (Spring Boot)   │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼              ┌─────────▼─────────┐              │
┌─────────────────┐     │                   │              │
│   Dịch Vụ       │◄────┤  Topology Mạng    │              │
│   Vệ Tinh       │     │                   │              │
└─────────────────┘     │                   │              │
         │               │                   │              │
         ▼               │                   │              │
┌─────────────────┐     │                   │              │
│ Dịch Vụ Trạm    │◄────┤                   │              │
│ Mặt Đất         │     │                   │              │
└─────────────────┘     │                   │              │
         │               │                   │              │
         ▼               │                   │              │
┌─────────────────┐     │                   │              │
│  Dịch Vụ        │◄────┤                   │              │
│  Router Lõi     │     └───────────────────┘              │
└─────────────────┘                                        │
         │                                                  │
         └──────────────────────────────────────────────────┘
```

## 🛠️ Tổng Quan Các Dịch Vụ

### 🟪 Client Service (JavaFX)
👉 **Vai trò**: Giao diện người dùng và tính năng chat với hiển thị mạng

**Tính năng mở rộng:**
- **Access Selector**: Chọn điểm truy cập vệ tinh/mặt đất (thủ công hoặc hỗ trợ AI)
- **Chat Session**: Tin nhắn đầu cuối với theo dõi gói tin
- **Topology Viewer**: Hiển thị mạng thời gian thực với trạng thái node
- **Log Manager**: Lịch sử trace gói tin và phân tích hiệu suất

**Cấu trúc code:**
```
client-service/
 ├─ domain/
 │   ├─ Packet.java
 │   ├─ Message.java
 │   └─ NodeInfo.java
 ├─ application/
 │   ├─ ChatSession.java
 │   ├─ AccessSelector.java
 │   └─ LogManager.java
 ├─ infrastructure/
 │   ├─ NetworkClient.java
 │   ├─ FirebaseLogger.java
 │   ├─ FirebaseReader.java
 │   └─ Config.java
 └─ interface/
     ├─ JavaFxApp.java
     ├─ ChatController.java
     └─ TopologyViewer.java
```

### 🟧 Satellite Service (Java)
👉 **Vai trò**: Mô phỏng node vệ tinh với chuyển tiếp gói tin quỹ đạo

**Tính năng mở rộng:**
- **QoS Module**: Mô phỏng delay/loss dựa trên điều kiện thời tiết
- **Buffering**: Xếp hàng gói tin khi trạm mặt đất quá tải
- **Fault Detection**: Báo cáo lỗi liên kết cho Manager Service
- **Orbital Mechanics**: Cập nhật vị trí dựa trên quỹ đạo vệ tinh

**Cấu trúc code:**
```
satellite-service/
 ├─ domain/
 │   ├─ Packet.java
 │   ├─ RoutingTable.java
 │   └─ LinkStatus.java
 ├─ application/
 │   ├─ Forwarder.java
 │   ├─ QoSModule.java
 │   ├─ BufferManager.java
 │   └─ StateManager.java
 ├─ infrastructure/
 │   ├─ TcpServer.java
 │   ├─ TcpSender.java
 │   ├─ FirebaseReporter.java
 │   └─ ManagerClient.java
 └─ interface/
     └─ SatelliteApp.java
```

### 🟨 Ground Station Service (Java)
👉 **Vai trò**: Gateway mặt đất giữa vệ tinh và mạng lõi

**Tính năng mở rộng:**
- **Bandwidth Limiter**: Thuật toán token bucket cho mô phỏng băng thông
- **Congestion Detection**: Cảnh báo Manager khi tải lưu lượng cao
- **Packet Scheduling**: Ưu tiên gói tin điều khiển hơn gói tin dữ liệu
- **Weather Adaptation**: Điều chỉnh tham số truyền dựa trên điều kiện

**Cấu trúc code:**
```
groundstation-service/
 ├─ domain/
 │   ├─ Packet.java
 │   ├─ RoutingTable.java
 │   └─ TrafficStat.java
 ├─ application/
 │   ├─ Forwarder.java
 │   ├─ BandwidthLimiter.java
 │   ├─ CongestionDetector.java
 │   └─ StateManager.java
 ├─ infrastructure/
 │   ├─ TcpServer.java
 │   ├─ TcpSender.java
 │   ├─ FirebaseReporter.java
 │   └─ ManagerClient.java
 └─ interface/
     └─ GroundStationApp.java
```

### 🟩 Core Router Service (Java)
👉 **Vai trò**: Hạ tầng định tuyến IP backbone dung lượng cao

**Tính năng mở rộng:**
- **Dynamic Routing Protocol**: Tính toán lại tuyến khi topology thay đổi
- **Load Balancing**: Phân phối gói tin qua nhiều đường dẫn
- **Failover**: Định tuyến lại tự động khi vệ tinh/trạm mặt đất lỗi
- **Traffic Engineering**: Tối ưu hóa sử dụng mạng

**Cấu trúc code:**
```
core-router-service/
 ├─ domain/
 │   ├─ Packet.java
 │   ├─ RoutingTable.java
 │   └─ LinkMetric.java
 ├─ application/
 │   ├─ Forwarder.java
 │   ├─ RoutingProtocol.java
 │   ├─ LoadBalancer.java
 │   └─ StateManager.java
 ├─ infrastructure/
 │   ├─ TcpServer.java
 │   ├─ TcpSender.java
 │   ├─ FirebaseReporter.java
 │   └─ ManagerClient.java
 └─ interface/
     └─ CoreRouterApp.java
```

### 🟦 Manager Service (Java, Spring Boot)
👉 **Vai trò**: Điều phối mạng cung cấp topology, thời tiết và API giám sát

**Tính năng mở rộng:**
- **Weather Generator**: Mô phỏng mưa, bão ảnh hưởng QoS vệ tinh
- **Topology Updater**: Định vị node động dựa trên quỹ đạo vệ tinh
- **Monitoring API**: Endpoint RESTful để truy vấn trạng thái toàn mạng
- **Firebase Synchronization**: Đồng bộ dữ liệu thời gian thực qua tất cả dịch vụ

**Cấu trúc code:**
```
manager-service/
 ├─ domain/
 │   ├─ NodeInfo.java
 │   ├─ Weather.java
 │   └─ Topology.java
 ├─ application/
 │   ├─ TopologyService.java
 │   ├─ WeatherService.java
 │   └─ MonitoringService.java
 ├─ infrastructure/
 │   ├─ FirebaseSync.java
 │   ├─ RandomWeatherGen.java
 │   └─ TopologyUpdater.java
 └─ interface/
     ├─ ManagerController.java  # REST API
     └─ ManagerApp.java
```

### 🟥 AI Optimizer Service (Python, FastAPI)
👉 **Vai trò**: Học tăng cường + AI sinh để lựa chọn tối ưu node truy cập và định tuyến

**Tính năng mở rộng:**
- **Hybrid AI**: RL cho tối ưu ngắn hạn, GenAI cho phân tích dài hạn
- **What-if Simulator**: Mô hình tác động của quyết định định tuyến thay thế
- **Recommendation API**: Truy vấn trực tiếp từ client để tư vấn định tuyến
- **Continuous Learning**: Thích ứng với điều kiện mạng thay đổi

**Cấu trúc code:**
```
ai-optimizer-service/
 ├─ domain/
 │   ├─ state.py
 │   ├─ action.py
 │   └─ reward.py
 ├─ application/
 │   ├─ rl_agent.py
 │   ├─ genai_advisor.py
 │   ├─ optimizer.py
 │   └─ simulator.py
 ├─ infrastructure/
 │   ├─ firebase_client.py
 │   ├─ api_client.py
 │   └─ config.py
 └─ interface/
     ├─ app.py             # FastAPI entrypoint
     └─ controllers.py
```

## 🔥 Schema Cơ Sở Dữ Liệu Firebase

Hệ thống sử dụng Firebase Realtime Database/Firestore để quản lý trạng thái thời gian thực và giám sát. Mỗi dịch vụ liên tục ghi trạng thái và log để AI phân tích và client quan sát.

### Các Collection Cơ Sở Dữ Liệu

#### 1. `logs/` - Log Lưu Lượng Mạng
Theo dõi tất cả gói tin đã đi qua mạng:
```json
logs/{logId}: {
  "packetId": "abc123",
  "src": "clientA",
  "dst": "clientB", 
  "path": ["clientA", "sat1", "ground1", "core1", "ground2", "sat2", "clientB"],
  "timestamp": 1693783456000,
  "status": "delivered",
  "delayMs": 123,
  "lossRate": 0.01
}
```

#### 2. `packets/` - Trạng Thái Gói Tin Hoạt Động
Trạng thái thời gian thực của gói tin hiện đang trong mạng:
```json
packets/{packetId}: {
  "src": "clientA",
  "dst": "clientB",
  "payloadSize": 512,
  "ttl": 5,
  "currentNode": "sat1",
  "nextHop": "ground1",
  "createdAt": 1693783456000
}
```

#### 3. `nodes/` - Thông Tin Node Mạng
Trạng thái trực tiếp của tất cả node mạng:
```json
nodes/{nodeId}: {
  "id": "sat1",
  "type": "satellite",
  "ip": "10.0.0.2",
  "coords": { "x": 1200, "y": 800, "z": 2000 },
  "status": "active",
  "bandwidth": 1000,
  "load": 0.65,
  "lastUpdated": 1693783456000
}
```

#### 4. `links/` - Trạng Thái Liên Kết Vật Lý
Kết nối mạng giữa các node:
```json
links/{linkId}: {
  "srcNode": "sat1",
  "dstNode": "ground1", 
  "latencyMs": 40,
  "lossRate": 0.001,
  "capacityMbps": 500,
  "status": "up"
}
```

#### 5. `topology/` - Snapshot Mạng
Trạng thái mạng hoàn chỉnh tại thời điểm cụ thể:
```json
topology/{snapshotId}: {
  "timestamp": 1693783456000,
  "nodes": ["clientA", "sat1", "ground1", "core1", "sat2", "clientB"],
  "links": ["link_sat1_ground1", "link_ground1_core1", "link_core1_sat2"]
}
```

#### 6. `weather/` - Điều Kiện Môi Trường
Tác động thời tiết lên liên kết vệ tinh/mặt đất:
```json
weather/{regionId}: {
  "region": "Asia-Pacific",
  "conditions": "rain",
  "lossFactor": 0.05,
  "latencyFactor": 1.2,
  "timestamp": 1693783456000
}
```

#### 7. `recommendations/` - Kết Quả Tối Ưu AI
Khuyến nghị định tuyến do AI tạo ra:
```json
recommendations/{sessionId}: {
  "client": "clientA",
  "dst": "clientB",
  "bestAccessNode": "sat1", 
  "confidence": 0.87,
  "reason": "low latency, stable link",
  "timestamp": 1693783456000
}
```

#### 8. `sessions/` - Phiên Chat
Quản lý phiên truyền thông:
```json
sessions/{sessionId}: {
  "participants": ["clientA", "clientB"],
  "messages": [
    { "from": "clientA", "text": "hello", "timestamp": 1693783400000 },
    { "from": "clientB", "text": "hi!", "timestamp": 1693783410000 }
  ],
  "status": "active"
}
```

### Mẫu Tương Tác Service-Database

| Service | Ghi Vào | Đọc Từ |
|---------|---------|--------|
| **Client Service** | `sessions/` | `recommendations/`, `logs/` |
| **Satellite Service** | `logs/`, `nodes/`, `links/` | `packets/`, `topology/` |
| **Ground Station** | `logs/`, `nodes/`, `links/` | `packets/`, `weather/` |
| **Core Router** | `logs/`, `nodes/`, `links/` | `packets/`, `topology/` |
| **Manager Service** | `topology/`, `weather/` | `nodes/`, `links/` |
| **AI Optimizer** | `recommendations/` | `nodes/`, `links/`, `logs/`, `weather/` |

### Luồng Dữ Liệu Thời Gian Thực
```
┌─────────────┐    ghi logs     ┌─────────────┐    đọc dữ liệu   ┌─────────────┐
│  Dịch Vụ    │ ──────────────► │  Firebase   │ ──────────────► │ AI Optimizer │
│   Mạng      │                 │  Database   │                 │   Service    │
└─────────────┘                 └─────────────┘                 └─────────────┘
       │                               │                               │
       │                               │ đọc recommendations          │
       │                               ◄───────────────────────────────┘
       │                               │
       │                               ▼
       │                        ┌─────────────┐
       └───────────────────────► │   Client    │
         lệnh mạng               │    GUI      │
                                └─────────────┘
```

## 🔄 Luồng Công Việc Mở Rộng

1. **Client** gửi gói tin → chọn node truy cập (thủ công hoặc AI khuyến nghị)
2. **Gói tin đi qua**: Vệ tinh → Trạm Mặt Đất → Router Lõi → Đích
3. **Mỗi node** ghi dữ liệu trạng thái/hiệu suất vào Firebase thời gian thực
4. **Manager** tạo cập nhật thời tiết/topology → phát sóng tới tất cả dịch vụ
5. **AI Optimizer** đọc dữ liệu mạng, huấn luyện mô hình RL, cập nhật khuyến nghị
6. **Client UI** hiển thị khuyến nghị thời gian thực + hiển thị topology trực tiếp
7. **Xử lý lỗi**: Lỗi mạng kích hoạt định tuyến lại tự động qua core routers

## 🚦 Bắt Đầu Nhanh

### Yêu Cầu Tiên Quyết
- Docker & Docker Compose
- Java 17+ (cho phát triển local)
- Python 3.11+ (cho phát triển dịch vụ AI)
- Tài khoản Firebase (cho tính năng giám sát)

### Chạy với Docker Compose

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd sagin-network
   ```

2. **Cấu hình thông tin xác thực Firebase**
   ```bash
   # Thêm cấu hình Firebase vào application.properties của mỗi dịch vụ
   cp resources/application.properties.template resources/application.properties
   # Chỉnh sửa với thông tin xác thực Firebase của bạn
   ```

3. **Build và khởi động tất cả dịch vụ**
   ```bash
   docker-compose up --build
   ```

4. **Truy cập các dịch vụ**
   - Manager API: http://localhost:8081
   - AI Optimizer API: http://localhost:8082
   - Client GUI: Sẽ khởi động tự động

### Chạy với Kubernetes

1. **Triển khai lên cluster Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

2. **Giám sát triển khai**
   ```bash
   kubectl get pods
   kubectl get services
   ```

## 📁 Cấu Trúc Dự Án

```
sagin-network/
├── client-service/           # Ứng dụng client JavaFX
├── satellite-service/        # Mô phỏng node vệ tinh
├── groundstation-service/    # Node trạm mặt đất  
├── core-router-service/     # Router mạng lõi
├── manager-service/         # API quản lý mạng
├── ai-optimizer-service/    # Engine tối ưu AI
├── docker-compose.yaml     # Điều phối multi-service
└── k8s/                    # Config triển khai Kubernetes
```

Mỗi service tuân theo mẫu kiến trúc sạch:
- **Domain**: Thực thể và logic nghiệp vụ cốt lõi
- **Application**: Use case và workflow nghiệp vụ  
- **Infrastructure**: Tích hợp bên ngoài (Firebase, TCP, v.v.)
- **Interface**: Controller API và điểm vào ứng dụng

## 🔧 Phát Triển

### Thiết Lập Phát Triển Local

1. **Dịch Vụ Java**
   ```bash
   cd <service-name>
   ./mvnw spring-boot:run
   # hoặc
   ./mvnw exec:java
   ```

2. **Dịch Vụ AI Python**
   ```bash
   cd ai-optimizer-service
   pip install -r requirements.txt
   uvicorn interface.app:app --reload
   ```

### Build Từng Dịch Vụ

Mỗi dịch vụ bao gồm script build:
```bash
cd <service-name>
./scripts/build.sh
./scripts/run.sh
```

## 🌐 API Endpoints

### Manager Service (Port 8081)
- `GET /api/topology` - Lấy topology mạng hiện tại
- `GET /api/nodes` - Liệt kê tất cả node mạng
- `POST /api/weather/update` - Cập nhật điều kiện thời tiết

### AI Optimizer Service (Port 8082)
- `POST /api/optimize` - Yêu cầu tối ưu định tuyến
- `GET /api/model/status` - Lấy trạng thái huấn luyện mô hình AI
- `POST /api/simulate` - Chạy mô phỏng mạng

## 📊 Giám Sát & Logging

Truy cập giám sát mạng thời gian thực qua:
- **Firebase Console**: Collection cơ sở dữ liệu thô và cập nhật thời gian thực
- **Manager Service Dashboard**: Thống kê mạng tổng hợp và xem topology
- **Client GUI**: Hiển thị mạng trực tiếp và log phiên chat
- **AI Optimizer Metrics**: Phân tích hiệu suất và khuyến nghị tối ưu

## 🤝 Đóng Góp

1. Fork repository
2. Tạo nhánh tính năng (`git checkout -b feature/tinh-nang-tuyet-voi`)
3. Commit thay đổi (`git commit -m 'Thêm tính năng tuyệt vời'`)
4. Push lên nhánh (`git push origin feature/tinh-nang-tuyet-voi`)
5. Mở Pull Request

## 📝 Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT - xem file [LICENSE](LICENSE) để biết chi tiết.

## 🆘 Hỗ Trợ

Để được hỗ trợ và câu hỏi:
- Tạo issue trong repository này
- Liên hệ nhóm phát triển
- Kiểm tra tài liệu trong README của mỗi dịch vụ

## 🔮 Lộ Trình Tương Lai

- [ ] Cải thiện mô hình machine learning
- [ ] Tính năng hiển thị nâng cao
- [ ] Hỗ trợ triển khai đa vùng
- [ ] Triển khai bảo mật nâng cao
- [ ] Công cụ benchmark hiệu suất

---
