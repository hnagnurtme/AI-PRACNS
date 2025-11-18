# MongoDB Connection Timeout - Giải pháp

## 🔍 Vấn đề

```
MongoSocketReadTimeoutException: Timeout while receiving message
Caused by: java.net.SocketTimeoutException: Read timed out
```

## 📊 Nguyên nhân có thể

### 1. **Network latency cao**
- MongoDB Atlas ở Singapore/US, app ở Vietnam
- ISP blocking/throttling MongoDB ports
- Network congestion

### 2. **MongoDB Atlas Free Tier (M0) limitations**
- Shared cluster, performance không đảm bảo
- Có thể "sleep" sau thời gian không hoạt động
- Connection limit thấp (max 100 connections cho free tier)

### 3. **Stale connections**
- Connections không được refresh
- Idle connections bị MongoDB server đóng
- Connection pool không được cấu hình đúng

### 4. **Change Stream specific issues**
- Change Streams giữ long-running connections
- Default timeout quá ngắn cho Change Streams
- Server heartbeat timeout

---

## ✅ Giải pháp đã áp dụng

### 1. **Cấu hình MongoDB Connection Settings**

File: [MongoConfiguration.java](src/SAGSINs/src/main/java/com/sagsins/core/configuration/MongoConfiguration.java)

```java
MongoClientSettings settings = MongoClientSettings.builder()
    .applyConnectionString(connectionString)

    // Socket timeout settings
    .applyToSocketSettings(builder ->
        builder.connectTimeout(10, TimeUnit.SECONDS)    // Timeout để connect
               .readTimeout(30, TimeUnit.SECONDS))       // Timeout để đọc data

    // Server selection timeout
    .applyToClusterSettings(builder ->
        builder.serverSelectionTimeout(15, TimeUnit.SECONDS))

    // Connection pool settings
    .applyToConnectionPoolSettings(builder ->
        builder.maxSize(50)                              // Max connections
               .minSize(5)                               // Min connections (keep-alive)
               .maxConnectionIdleTime(60, TimeUnit.SECONDS)   // Close idle connections
               .maxConnectionLifeTime(120, TimeUnit.SECONDS)) // Refresh stale connections
    .build();
```

**Giải thích:**

- **connectTimeout (10s):** Thời gian chờ để thiết lập kết nối TCP
- **readTimeout (30s):** Thời gian chờ để nhận response từ MongoDB
  - Quan trọng cho Change Streams vì chúng giữ connection lâu
  - Default là 0 (infinite), nhưng network issues có thể gây hang
- **serverSelectionTimeout (15s):** Thời gian chờ để tìm MongoDB server
- **maxSize (50):** Tối đa 50 connections trong pool
- **minSize (5):** Luôn giữ 5 connections sẵn sàng
- **maxConnectionIdleTime (60s):** Đóng connections idle sau 60s
- **maxConnectionLifeTime (120s):** Refresh connections sau 2 phút

### 2. **Improved Error Handling**

File: [PacketChangeStreamService.java](src/SAGSINs/src/main/java/com/sagsins/core/service/PacketChangeStreamService.java)

```java
} catch (IllegalArgumentException e) {
    // Enum parsing error
    logger.error("❌ [ENUM ERROR] Failed to parse TwoPacket due to enum mismatch: {}", e.getMessage());
    logger.error("   - This is likely due to serviceType enum mismatch");
} catch (Exception e) {
    // Other errors
    logger.error("❌ [ERROR] Error handling TwoPacket change: {}", e.getMessage(), e);
}
```

**Benefits:**
- Distinguish enum errors from network errors
- Change Stream không crash khi có bad data
- Detailed logging để debug

---

## 🧪 Testing

### Test 1: Verify connection settings

```bash
# Start Java application and check logs
# Should see:
# - "Connected to MongoDB"
# - "✅ MessageListenerContainer is RUNNING"
# - No timeout errors
```

### Test 2: Test Change Stream với Python

```bash
cd /Users/anhnon/PBL4
python3 test_change_stream.py
```

**Expected behavior:**
- Insert → Java nhận INSERT event sau vài giây
- Replace → Java nhận REPLACE event
- Không có timeout errors

### Test 3: Monitor connection pool

Thêm logging để monitor connection pool (optional):

```java
logger.info("Connection pool stats: {}", mongoClient.getClusterDescription());
```

---

## 🔧 Nếu vẫn gặp timeout

### Solution 1: Tăng timeout values

```java
.readTimeout(60, TimeUnit.SECONDS)  // Tăng từ 30s → 60s
```

### Solution 2: Add retry logic

```java
@Retryable(
    value = {MongoSocketReadTimeoutException.class},
    maxAttempts = 3,
    backoff = @Backoff(delay = 1000)
)
public void handleChange(...) {
    // Change stream handler
}
```

### Solution 3: Upgrade MongoDB Atlas

- Free tier M0 → M2/M10
- Dedicated cluster, better performance
- Higher connection limits
- Guaranteed resources

### Solution 4: Use local MongoDB for development

```bash
# Run local MongoDB with replica set
docker-compose up -d mongodb
```

Update connection string:
```java
String uri = "mongodb://localhost:27017/network?replicaSet=rs0";
```

### Solution 5: Implement heartbeat monitoring

```java
@Scheduled(fixedDelay = 30000)
public void checkMongoHealth() {
    try {
        mongoTemplate.executeCommand("{ ping: 1 }");
        logger.info("✅ MongoDB connection healthy");
    } catch (Exception e) {
        logger.error("❌ MongoDB connection unhealthy: {}", e.getMessage());
        // Restart connection if needed
    }
}
```

---

## 📊 Monitoring

### Metrics to track:

1. **Connection pool stats:**
   - Active connections
   - Idle connections
   - Wait queue size

2. **Change Stream health:**
   - Number of events received per minute
   - Error rate
   - Event processing time

3. **Network metrics:**
   - Ping latency to MongoDB Atlas
   - Packet loss rate
   - Connection failures

### Recommended tools:

- **Spring Boot Actuator:** Monitor application health
- **Prometheus + Grafana:** Metrics visualization
- **MongoDB Atlas Monitoring:** Built-in monitoring dashboard

---

## 🎯 Tóm tắt

### Vấn đề:
- MongoDB timeout khi Change Stream đang hoạt động
- Network latency hoặc MongoDB Atlas free tier limitations

### Giải pháp:
- ✅ Configured proper timeouts (connect, read, server selection)
- ✅ Set up connection pool với min/max sizes
- ✅ Refresh stale connections (maxConnectionLifeTime)
- ✅ Better error handling (distinguish enum vs network errors)

### Next steps:
- Monitor connection health
- Consider upgrading MongoDB tier nếu timeout vẫn xảy ra thường xuyên
- Implement retry logic nếu cần
- Use local MongoDB for development

---

## 🔗 Related Files

- [MongoConfiguration.java](src/SAGSINs/src/main/java/com/sagsins/core/configuration/MongoConfiguration.java)
- [PacketChangeStreamService.java](src/SAGSINs/src/main/java/com/sagsins/core/service/PacketChangeStreamService.java)
- [ServiceType.java](src/SAGSINs/src/main/java/com/sagsins/core/model/ServiceType.java)
- [test_change_stream.py](test_change_stream.py)
