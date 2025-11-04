# Scheduled Packet Sender - Hướng dẫn sử dụng

## Tổng quan
Hệ thống đã được chuyển đổi từ cơ chế POST API sang **cơ chế tự động schedule** để đọc và gửi dữ liệu packet từ MongoDB collections.

## Các thay đổi chính

### 1. Files mới được tạo

#### Repositories
- **`ITwoPacketRepository.java`**: Repository để truy cập collection `two_packets`
- **`IBatchPacketRepository.java`**: Repository để truy cập collection `batch_packets`

#### Service
- **`PacketSchedulerService.java`**: Service chứa các scheduled tasks để tự động đọc và gửi packets

### 2. Files được cập nhật

#### CoreApplication.java
- Thêm `@EnableScheduling` để kích hoạt chức năng scheduling trong Spring Boot

#### PacketController.java
- **Đã xóa**: 
  - `POST /api/v1/packets/double` 
  - `POST /api/v1/packets/batch`
- **Giữ lại**: `POST /api/v1/packets` cho single packet

#### Model Classes
- **TwoPacket.java**: Thêm `@Document(collection = "two_packets")`
- **BatchPacket.java**: Thêm `@Document(collection = "batch_packets")`

## Cách hoạt động

### Scheduled Tasks

#### 1. TwoPacket Sender
```java
@Scheduled(fixedRate = 5000) // Chạy mỗi 5 giây
public void sendTwoPackets()
```
- **Tần suất**: Mỗi 5 giây
- **Collection**: `two_packets`
- **Topic WebSocket**: `/topic/packets`
- **Chức năng**: 
  - Đọc tất cả documents từ collection `two_packets`
  - Gửi từng TwoPacket qua WebSocket
  - Log thông tin chi tiết về pairId, dijkstraPacket, rlPacket

#### 2. BatchPacket Sender
```java
@Scheduled(fixedRate = 10000) // Chạy mỗi 10 giây
public void sendBatchPackets()
```
- **Tần suất**: Mỗi 10 giây
- **Collection**: `batch_packets`
- **Topic WebSocket**: `/topic/batchpacket`
- **Chức năng**: 
  - Đọc tất cả documents từ collection `batch_packets`
  - Gửi từng BatchPacket qua WebSocket
  - Log thông tin về batchId, totalPairPackets, số lượng packets

## Cấu trúc Database

### Collection: `two_packets`
```json
{
  "_id": ObjectId,
  "pairId": "sourceUserId_destinationUserId",
  "dijkstraPacket": {
    "packetId": "string",
    "sourceUserId": "string",
    "destinationUserId": "string",
    ...
  },
  "rlPacket": {
    "packetId": "string",
    "sourceUserId": "string",
    "destinationUserId": "string",
    ...
  }
}
```

### Collection: `batch_packets`
```json
{
  "_id": ObjectId,
  "batchId": "sourceUserId_destinationUserId",
  "totalPairPackets": 10,
  "packets": [
    {
      "pairId": "user1_user2",
      "dijkstraPacket": {...},
      "rlPacket": {...}
    },
    ...
  ]
}
```

## Cấu hình tùy chỉnh

### Thay đổi tần suất Schedule

Trong file `PacketSchedulerService.java`, bạn có thể điều chỉnh:

#### Tùy chọn 1: Fixed Rate (khoảng thời gian cố định)
```java
@Scheduled(fixedRate = 5000) // milliseconds
```

#### Tùy chọn 2: Fixed Delay (delay cố định sau mỗi lần chạy)
```java
@Scheduled(fixedDelay = 5000) // milliseconds
```

#### Tùy chọn 3: Cron Expression (lịch cụ thể)
```java
@Scheduled(cron = "0 */5 * * * *") // Mỗi 5 phút
```

### Ví dụ Cron Expressions
- `"0 */1 * * * *"` - Mỗi 1 phút
- `"0 0 * * * *"` - Mỗi giờ
- `"0 0 0 * * *"` - Mỗi ngày lúc 00:00
- `"0 0 9-17 * * MON-FRI"` - Mỗi giờ từ 9h-17h, thứ 2-6

## Testing

### 1. Thêm dữ liệu test vào MongoDB

```javascript
// Thêm vào collection two_packets
db.two_packets.insertOne({
  "pairId": "user1_user2",
  "dijkstraPacket": {
    "packetId": "PKT001",
    "sourceUserId": "user1",
    "destinationUserId": "user2",
    "type": "DATA",
    "payloadSizeByte": 1024,
    "isUseRL": false
  },
  "rlPacket": {
    "packetId": "PKT002",
    "sourceUserId": "user1",
    "destinationUserId": "user2",
    "type": "DATA",
    "payloadSizeByte": 1024,
    "isUseRL": true
  }
})

// Thêm vào collection batch_packets
db.batch_packets.insertOne({
  "batchId": "batch_001",
  "totalPairPackets": 2,
  "packets": [
    {
      "pairId": "user1_user2",
      "dijkstraPacket": { "packetId": "PKT001", ... },
      "rlPacket": { "packetId": "PKT002", ... }
    },
    {
      "pairId": "user3_user4",
      "dijkstraPacket": { "packetId": "PKT003", ... },
      "rlPacket": { "packetId": "PKT004", ... }
    }
  ]
})
```

### 2. Xem logs

Sau khi chạy application, bạn sẽ thấy logs như:
```
INFO  c.s.c.s.PacketSchedulerService : Sent TwoPacket: pairId=user1_user2, dijkstra=PKT001, rl=PKT002
INFO  c.s.c.s.PacketSchedulerService : Successfully sent 1 TwoPackets
INFO  c.s.c.s.PacketSchedulerService : Sent BatchPacket: batchId=batch_001, totalPairs=2, packetsCount=2
INFO  c.s.c.s.PacketSchedulerService : Successfully sent 1 BatchPackets
```

### 3. Subscribe WebSocket để nhận data

Sử dụng client WebSocket để subscribe:
- `/topic/packets` - Nhận TwoPacket
- `/topic/batchpacket` - Nhận BatchPacket

## Lợi ích của cơ chế Scheduled

1. **Tự động hóa**: Không cần gọi API thủ công
2. **Đồng bộ**: Đảm bảo dữ liệu được gửi định kỳ
3. **Scalable**: Dễ dàng thêm nhiều scheduled tasks
4. **Monitoring**: Logs chi tiết cho việc theo dõi
5. **Flexible**: Dễ dàng điều chỉnh tần suất gửi

## Troubleshooting

### Không thấy packets được gửi
1. Kiểm tra collections `two_packets` và `batch_packets` có dữ liệu không
2. Xem logs để kiểm tra có errors không
3. Đảm bảo `@EnableScheduling` đã được thêm vào `CoreApplication`

### Gửi quá nhiều packets
1. Giảm `fixedRate` trong `@Scheduled` annotation
2. Thêm pagination khi query database
3. Thêm điều kiện filter để chỉ gửi packets cần thiết

### Memory issues
1. Thêm pagination để không load tất cả documents cùng lúc
2. Thêm cleanup task để xóa packets đã gửi
3. Implement caching mechanism

## Liên hệ
Nếu có vấn đề, vui lòng tạo issue hoặc liên hệ team.
