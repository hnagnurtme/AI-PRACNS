# MongoDB Change Stream Debug Guide

## 🔍 Vấn đề

Java `PacketChangeStreamService` không nhận được tín hiệu từ MongoDB khi Python `BatchPacketService` lưu packet thành công.

---

## 📊 Phân tích vấn đề

### ✅ Những gì đã ĐÚNG:

1. **Connection String:** Cả Java và Python đều kết nối đến cùng MongoDB cluster:
   - URI: `mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/`
   - Database: `network`

2. **Collection Names:** Khớp nhau
   - Python: `two_packets`, `batch_packets`
   - Java: `two_packets`, `batch_packets`

3. **Operations:** Python sử dụng `replace_one()` tạo ra REPLACE events
   - Java đang lắng nghe: `insert`, `update`, `replace` ✅

4. **MongoDB Atlas:** Mặc định hỗ trợ Change Streams (Replica Set) ✅

### ❌ Các nguyên nhân có thể:

1. **Java service chưa chạy hoặc chưa start MessageListenerContainer**
2. **Lỗi khi khởi tạo Change Stream listeners (bị catch và silent)**
3. **MongoDB Change Streams chưa được enable trên cluster**
4. **Network/firewall issues giữa Java app và MongoDB**
5. **Spring Boot context chưa load @PostConstruct**

---

## 🛠️ Cách kiểm tra và fix

### Bước 1: Kiểm tra Java Service có chạy không

Khi khởi động Java application, kiểm tra logs cho các dòng sau:

```
Initializing MongoDB Change Stream listeners...
📊 MongoDB Connection Info:
   - Database: network
   - Collections: two_packets, batch_packets
✅ Created scheduler for packet sending
✅ Created scheduler for packet deletion
✅ Registered Change Stream listener for 'two_packets' collection
✅ Registered Change Stream listener for 'batch_packets' collection
🚀 Starting MessageListenerContainer...
✅ MessageListenerContainer is RUNNING
✅ MongoDB Change Stream listeners started successfully
🎯 Ready to receive change events from MongoDB
```

**Nếu không thấy logs trên:**
- Java service chưa chạy
- Hoặc `PacketChangeStreamService` bean chưa được Spring Boot load
- Kiểm tra `@Service` annotation có đúng không
- Kiểm tra component scan có bao gồm package này không

**Nếu thấy lỗi:**
```
❌ Failed to initialize Change Stream listeners: ...
```
- Đọc error message để biết nguyên nhân
- Có thể là MongoDB connection issue
- Hoặc Change Streams không được enable

---

### Bước 2: Chạy test script để verify

Đã tạo test script: `test_change_stream.py`

**Cách chạy:**

```bash
cd /Users/anhnon/PBL4
python3 test_change_stream.py
```

**Script sẽ:**
1. Insert một TwoPacket vào MongoDB
2. Đợi 5 giây (để Java nhận INSERT event)
3. Replace TwoPacket (update thêm RL packet)
4. Đợi 5 giây (để Java nhận REPLACE event)
5. Cleanup test data

**Trong khi chạy, kiểm tra Java logs:**

Nếu Change Stream hoạt động, sẽ thấy:
```
🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: INSERT
🔄 [INSERT] TwoPacket received - pairId=USER_HANOI_USER_BANGKOK
⏰ Scheduled TwoPacket send in 3000ms - pairId=USER_HANOI_USER_BANGKOK
```

Sau khi replace:
```
🔔 [CHANGE EVENT] Received change event for two_packets collection!
📝 Operation Type: REPLACE
🔄 [REPLACE] TwoPacket received - pairId=USER_HANOI_USER_BANGKOK
⏰ Scheduled TwoPacket send in 3000ms - pairId=USER_HANOI_USER_BANGKOK
📤 [SENT] TwoPacket to /topic/packets
```

**Nếu KHÔNG thấy logs trên:**
- Change Stream không hoạt động
- Xem bước 3 và 4

---

### Bước 3: Verify MongoDB Change Streams enabled

**Kiểm tra trên MongoDB Atlas:**

1. Login vào https://cloud.mongodb.com/
2. Chọn cluster `mongo1`
3. Vào **Database Access** → Kiểm tra user `admin` có quyền read/write
4. Vào **Cluster** → **Configuration**:
   - Kiểm tra cluster là **Replica Set** (không phải Standalone)
   - Cluster tier phải >= M10 (Free tier M0 KHÔNG hỗ trợ Change Streams)

**⚠️ QUAN TRỌNG:**
MongoDB Atlas **Free Tier (M0)** KHÔNG hỗ trợ Change Streams!
- Cần upgrade lên ít nhất **M2** ($9/month) hoặc **M10** ($57/month)

**Cách kiểm tra tier:**
- Vào MongoDB Atlas Dashboard
- Cluster name sẽ hiển thị tier (ví dụ: "mongo1 - M0", "mongo1 - M10")

---

### Bước 4: Test Change Stream trực tiếp với MongoDB

**Sử dụng MongoDB Compass hoặc mongosh:**

```javascript
// Kết nối với mongosh
mongosh "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/network"

// Mở change stream
use network
db.two_packets.watch()

// Trong terminal khác, insert một document
use network
db.two_packets.insertOne({
  pairId: "TEST_PAIR",
  dijkstraPacket: { packetId: "TEST_001" },
  rlPacket: null
})
```

**Nếu Change Stream hoạt động:**
- Terminal đầu tiên sẽ in ra change event
- Format: `{ operationType: 'insert', fullDocument: {...} }`

**Nếu KHÔNG thấy change event:**
- MongoDB không hỗ trợ Change Streams
- Có thể do cluster tier quá thấp (M0)

---

### Bước 5: Nếu vẫn không hoạt động

**Giải pháp thay thế: Polling thay vì Change Streams**

Thay vì dùng Change Streams, có thể:

1. **Option 1: Scheduled Polling**
   - Java service query MongoDB mỗi 1-2 giây
   - Lấy documents mới/updated dựa vào timestamp
   - Đơn giản hơn nhưng tốn tài nguyên hơn

2. **Option 2: WebSocket/REST API**
   - Python service gọi API của Java service sau khi lưu packet
   - Java service nhận HTTP request và xử lý
   - Reliable hơn nhưng cần thêm code

3. **Option 3: Message Queue (RabbitMQ/Kafka)**
   - Python publish message khi lưu packet
   - Java consume message từ queue
   - Professional solution nhưng phức tạp hơn

**Ví dụ Scheduled Polling:**

```java
@Scheduled(fixedDelay = 2000) // 2 seconds
public void pollForNewPackets() {
    // Query two_packets với timestamp > lastCheck
    // Process new/updated packets
}
```

---

## 🎯 Checklist Debug

- [ ] Java service đang chạy
- [ ] Thấy logs "✅ MessageListenerContainer is RUNNING"
- [ ] Chạy `test_change_stream.py` thành công
- [ ] Thấy logs "🔔 [CHANGE EVENT]" trong Java
- [ ] MongoDB cluster là Replica Set (không phải Standalone)
- [ ] MongoDB tier >= M2 (không phải M0 free tier)
- [ ] Test change stream trực tiếp với mongosh thành công

---

## 📝 Improved Code Changes

### Đã cải thiện:

1. **PacketChangeStreamService.java:**
   - ✅ Thêm detailed logging khi start
   - ✅ Verify MessageListenerContainer is running
   - ✅ Log mỗi change event nhận được
   - ✅ Log operation type (INSERT/REPLACE/UPDATE)

2. **test_change_stream.py:**
   - ✅ Script để test Change Stream hoạt động
   - ✅ Insert → Wait → Replace → Wait → Cleanup
   - ✅ Hướng dẫn kiểm tra Java logs

---

## 🚀 Kết luận

**Nếu sau khi làm theo tất cả các bước trên mà vẫn không hoạt động:**

→ **Nguyên nhân chắc chắn:** MongoDB Free Tier (M0) không hỗ trợ Change Streams

**Giải pháp:**
1. Upgrade MongoDB Atlas lên M2/M10
2. Hoặc sử dụng alternative approach (Polling/API/Message Queue)
3. Hoặc deploy local MongoDB Replica Set cho development

**Recommended:** Sử dụng Polling approach cho development, Change Streams cho production.
