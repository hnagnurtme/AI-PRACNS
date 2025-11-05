# 🔧 Hướng dẫn Debug vấn đề gửi Packet

## ✅ Các cải tiến đã thực hiện

### 1. **Kiểm tra kết nối trước khi gửi**
- Thêm kiểm tra `NetworkUtils.isServiceAvailable()` trước khi gửi packet
- Hiển thị thông báo rõ ràng nếu server không phản hồi

### 2. **Thêm field `type` cho Packet**
- Set `type = "DATA"` cho mỗi packet
- Server có thể yêu cầu field này

### 3. **Cải thiện logging**
- Thêm emoji để dễ nhìn: 📤 gửi, ✅ thành công, ❌ lỗi
- Log chi tiết mỗi bước: kiểm tra server, gửi packet, kết quả

### 4. **Cải thiện PacketSender**
- Kiểm tra socket còn mở trước khi sử dụng lại
- Thêm timeout cho connect và read (5 giây)
- Set `keepAlive` để giữ kết nối ổn định
- Validate input: host, port, packet

### 5. **Xử lý lỗi tốt hơn**
- Exception messages chi tiết hơn
- Stack trace đầy đủ để debug

## 🐛 Các nguyên nhân thường gặp

### 1. Server không chạy
```
❌ Lỗi: Không thể kết nối đến server <host>:<port>. Server có đang chạy không?
```
**Giải pháp:** Kiểm tra server đã khởi động chưa, đúng IP/port chưa

### 2. IP/Port không đúng trong MongoDB
```
Lỗi: Trạm nguồn <stationId> thiếu thông tin IP/Port
```
**Giải pháp:** Kiểm tra collection `nodes` trong MongoDB, đảm bảo field `communication.ipAddress` và `communication.port` có giá trị

### 3. Firewall chặn kết nối
```
❌ Failed to create connection to <host>:<port>: Connection refused
```
**Giải pháp:** 
- Tắt firewall hoặc mở port
- Kiểm tra server có bind đúng interface không (0.0.0.0 vs 127.0.0.1)

### 4. Socket đã đóng
```
⚠️ Detected closed socket for <host>:<port>, recreating...
```
**Giải pháp:** Code đã tự động xử lý, sẽ tạo lại kết nối

### 5. Serialization lỗi
```
Failed to serialize packet with ID: <packetId>
```
**Giải pháp:** Kiểm tra model `Packet` có field nào không serialize được không

## 📋 Checklist Debug

Khi gửi packet thất bại, hãy kiểm tra theo thứ tự:

1. ✅ **MongoDB có chạy không?**
   ```bash
   mongo --eval "db.adminCommand('ping')"
   ```

2. ✅ **Dữ liệu trong MongoDB có đúng không?**
   ```javascript
   db.users.find({})
   db.nodes.find({})
   // Kiểm tra field communication.ipAddress và communication.port
   ```

3. ✅ **Server có đang chạy không?**
   ```bash
   # Kiểm tra port có đang được lắng nghe không
   lsof -i :<port>
   # hoặc
   netstat -an | grep <port>
   ```

4. ✅ **Có thể kết nối đến server không?**
   ```bash
   telnet <host> <port>
   # hoặc
   nc -zv <host> <port>
   ```

5. ✅ **Xem log trong console**
   - Log sẽ hiển thị từng bước: kiểm tra server, gửi packet, kết quả
   - Tìm các message với emoji: 📤, ✅, ❌, 🔌, ⚠️

6. ✅ **Kiểm tra exception stack trace**
   - Tất cả exception đều được print ra console
   - Đọc kỹ message để biết nguyên nhân

## 💡 Tips

### Kiểm tra kết nối thủ công
```java
boolean available = NetworkUtils.isServiceAvailable("192.168.1.100", 8080, 2000);
System.out.println("Server available: " + available);
```

### Test gửi 1 packet đơn giản
- Set packet count = 1
- Xem log chi tiết

### Kiểm tra network interface
```bash
# macOS/Linux
ifconfig
# hoặc
ip addr show

# Đảm bảo IP trong MongoDB khớp với IP của server
```

### Tắt firewall tạm thời (chỉ để test)
```bash
# macOS
sudo pfctl -d

# Enable lại
sudo pfctl -e
```

## 📝 Log mẫu khi thành công

```
🔍 Đang kiểm tra kết nối đến 192.168.1.100:8080...
Checking service availability on 192.168.1.100:8080
✅ Server 192.168.1.100:8080 đã sẵn sàng!
🔌 Creating new persistent connection to 192.168.1.100:8080
✅ Successfully connected to 192.168.1.100:8080
📤 Gửi RL packet: abc-123 -> 192.168.1.100:8080
✅ Đã gửi RL packet: abc-123
📤 Gửi non-RL packet: abc-123 -> 192.168.1.100:8080
✅ Đã gửi non-RL packet: abc-123
```

## 📝 Log mẫu khi thất bại

```
🔍 Đang kiểm tra kết nối đến 192.168.1.100:8080...
Checking service availability on 192.168.1.100:8080
❌ Lỗi: Không thể kết nối đến server 192.168.1.100:8080. Server có đang chạy không?
❌ Server 192.168.1.100:8080 không phản hồi!
```

## 🔄 Các bước tiếp theo

1. Chạy lại ứng dụng: `mvn javafx:run`
2. Xem log trong console
3. Thử gửi 1 packet
4. Nếu vẫn lỗi, copy log và phân tích theo checklist trên
