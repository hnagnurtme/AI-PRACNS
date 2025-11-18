# MongoDB Change Stream - Tổng kết vấn đề và giải pháp

## 🔍 Vấn đề ban đầu

Java `PacketChangeStreamService` không nhận được tín hiệu từ MongoDB khi Python `BatchPacketService` lưu packet thành công.

---

## ✅ Kết quả phân tích

### **Change Stream ĐÃ HOẠT ĐỘNG!**

Sau khi thêm logging chi tiết, phát hiện:
- ✅ MessageListenerContainer đã start thành công
- ✅ Change events ĐÃ được nhận từ MongoDB
- ✅ Python write operations ĐÃ trigger change events

**Log chứng minh:**
```
🔔 [CHANGE EVENT] Received change event for batch_packets collection!
```

---

## ❌ Vấn đề thật sự: ENUM MISMATCH

### Lỗi:
```
No enum constant com.sagsins.core.model.ServiceType.VIDEO_STREAMING
```

### Nguyên nhân:

**Python (Packet.py) sử dụng:**
```python
service_qos = QoS(
    service_type="VIDEO_STREAMING",  # ❌ Với "ING"
    ...
)
```

**Java enum ServiceType có:**
```java
VIDEO_STREAM,  // ❌ Không có "ING"
AUDIO_CALL,
IMAGE_TRANSFER,
TEXT_MESSAGE,
FILE_TRANSFER
```

Khi Spring Data MongoDB cố gắng deserialize document từ Change Stream event, nó không tìm thấy enum value `VIDEO_STREAMING` → **throw exception** → Change event bị dropped!

---

## 🛠️ Giải pháp đã áp dụng

### 1. **Fix enum ServiceType.java**

Thêm support cho cả hai variants:

```java
public enum ServiceType {
    @JsonProperty("VIDEO_STREAMING")
    VIDEO_STREAM,

    @JsonProperty("AUDIO_CALL")
    AUDIO_CALL,

    @JsonProperty("IMAGE_TRANSFER")
    IMAGE_TRANSFER,

    @JsonProperty("TEXT_MESSAGE")
    TEXT_MESSAGE,

    @JsonProperty("FILE_TRANSFER")
    FILE_TRANSFER,

    // Backward compatibility - also accept VIDEO_STREAMING as enum value
    VIDEO_STREAMING
}
```

**Cách hoạt động:**
- `@JsonProperty("VIDEO_STREAMING")` cho phép deserialize từ "VIDEO_STREAMING" thành `VIDEO_STREAM`
- Thêm `VIDEO_STREAMING` như một enum constant riêng để backward compatibility
- Giờ cả `VIDEO_STREAMING` và `VIDEO_STREAM` đều được accept

### 2. **Improved logging trong PacketChangeStreamService.java**

Thêm detailed logs để debug dễ hơn:

```java
@PostConstruct
public void initChangeStreamListeners() {
    logger.info("Initializing MongoDB Change Stream listeners...");
    logger.info("📊 MongoDB Connection Info:");
    logger.info("   - Database: {}", mongoTemplate.getDb().getName());
    logger.info("   - Collections: two_packets, batch_packets");

    // ... scheduler initialization ...

    logger.info("🚀 Starting MessageListenerContainer...");
    messageListenerContainer.start();

    // Verify container is running
    if (messageListenerContainer.isRunning()) {
        logger.info("✅ MessageListenerContainer is RUNNING");
    } else {
        logger.warn("⚠️ MessageListenerContainer is NOT running!");
    }

    logger.info("🎯 Ready to receive change events from MongoDB");
}
```

Thêm logs khi nhận change events:

```java
private void handleTwoPacketChange(...) {
    logger.info("🔔 [CHANGE EVENT] Received change event for two_packets collection!");

    // Extract operation type
    String operationType = "unknown";
    if (raw != null && raw.getOperationType() != null) {
        operationType = raw.getOperationType().getValue();
    }
    logger.info("📝 Operation Type: {}", operationType.toUpperCase());

    // ... rest of processing ...
}
```

### 3. **Created test script: test_change_stream.py**

Script để test Change Stream hoạt động:
- Insert TwoPacket
- Wait 5 seconds
- Replace (update) TwoPacket
- Wait 5 seconds
- Cleanup

**Cách chạy:**
```bash
cd /Users/anhnon/PBL4
python3 test_change_stream.py
```

---

## 📊 Tóm tắt timeline phân tích

1. **Ban đầu:** Nghĩ rằng Change Stream không hoạt động
2. **Phân tích:** Kiểm tra connection, collections, operations → Tất cả đều đúng
3. **Added logging:** Phát hiện Change Stream **ĐÃ hoạt động**
4. **Root cause:** Enum mismatch `VIDEO_STREAMING` vs `VIDEO_STREAM`
5. **Fixed:** Thêm support cho cả hai variants

---

## ✅ Kết quả sau khi fix

Sau khi restart Java application với code mới:

1. **Python save packet:**
   ```python
   service.save_packet(packet)  # packet.service_qos.service_type = "VIDEO_STREAMING"
   ```

2. **MongoDB:**
   - Document được lưu vào `two_packets` collection
   - Change Stream event được trigger

3. **Java nhận event:**
   ```
   🔔 [CHANGE EVENT] Received change event for two_packets collection!
   📝 Operation Type: REPLACE
   ⏰ Scheduled TwoPacket send in 3000ms
   📤 [SENT] TwoPacket to /topic/packets
   ```

4. **WebSocket:**
   - Frontend nhận message qua `/topic/packets`
   - Hiển thị packet comparison

---

## 🎯 Các file đã thay đổi

1. **[ServiceType.java](src/SAGSINs/src/main/java/com/sagsins/core/model/ServiceType.java)**
   - Thêm `@JsonProperty` annotations
   - Thêm `VIDEO_STREAMING` enum constant

2. **[PacketChangeStreamService.java](src/SAGSINs/src/main/java/com/sagsins/core/service/PacketChangeStreamService.java)**
   - Improved startup logging
   - Added change event logging
   - Added operation type logging

3. **[test_change_stream.py](test_change_stream.py)**
   - Test script để verify Change Stream hoạt động

4. **[CHANGESTREAM_DEBUG_GUIDE.md](CHANGESTREAM_DEBUG_GUIDE.md)**
   - Hướng dẫn debug và troubleshooting

---

## 💡 Bài học

1. **Logging is crucial:** Detailed logs giúp phát hiện vấn đề nhanh chóng
2. **Test early:** Script test đơn giản giúp verify functionality
3. **Data contract:** Python và Java phải sync về enum values, field names
4. **Error handling:** Silent errors (caught exceptions) rất khó debug

---

## 🚀 Next steps (nếu cần)

1. **Standardize enum values:** Quyết định dùng `VIDEO_STREAM` hay `VIDEO_STREAMING`
   - Update Python code để match với Java
   - Hoặc giữ nguyên và dùng `@JsonProperty` mapping

2. **Add validation:** Validate serviceType trước khi save vào MongoDB
   - Đảm bảo chỉ dùng các giá trị hợp lệ

3. **Monitor Change Stream health:**
   - Add metrics để track số change events received
   - Alert nếu không nhận được events trong X phút

4. **Consider DTOs:** Tách riêng MongoDB models và API DTOs
   - Flexible hơn khi thay đổi database schema
