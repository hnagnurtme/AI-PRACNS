# Terminal Controls - Hướng dẫn sử dụng

## 📡 Terminal Controls là gì?

**Terminal Controls** là một bảng điều khiển trong Dashboard cho phép bạn quản lý **User Terminals** (thiết bị đầu cuối người dùng) trong mạng SAGIN.

## 🎯 Tác dụng chính

### 1. **Tạo Terminals (Generate Terminals)**
- **Mục đích**: Tạo các thiết bị đầu cuối người dùng mới để mô phỏng nhu cầu kết nối trong mạng SAGIN
- **Cách hoạt động**:
  - Nhập số lượng terminals muốn tạo (1-100)
  - Click "Generate Terminals"
  - Hệ thống sẽ tạo các terminals với:
    - Vị trí ngẫu nhiên trên bản đồ
    - Các loại khác nhau: MOBILE, FIXED, VEHICLE, AIRCRAFT
    - QoS requirements (yêu cầu chất lượng dịch vụ) ngẫu nhiên
    - Trạng thái ban đầu: `idle` (chưa kết nối)

### 2. **Xóa Terminals (Clear)**
- **Mục đích**: Xóa tất cả terminals hiện có để bắt đầu lại từ đầu
- **Sử dụng khi**: Muốn reset môi trường test hoặc tạo scenario mới

### 3. **Theo dõi Trạng thái (Terminal Status)**
Hiển thị thống kê real-time về terminals:
- **Total**: Tổng số terminals
- **Idle**: Số terminals chưa kết nối
- **Connected**: Số terminals đã kết nối với nodes
- **Transmitting**: Số terminals đang truyền dữ liệu

## 🔗 Mối quan hệ với hệ thống

### Terminals trong mạng SAGIN

```
User Terminals → Kết nối với → Nodes (Satellites/Ground Stations) → Truyền dữ liệu
```

**Terminals** đại diện cho:
- 📱 **Thiết bị người dùng cuối**: Điện thoại, máy tính, thiết bị IoT
- 🚗 **Phương tiện**: Xe tự lái, máy bay, tàu thủy
- 🏢 **Trạm cố định**: Trạm quan sát, trạm nghiên cứu

### Quy trình hoạt động

1. **Tạo Terminals** (Terminal Controls)
   - Tạo các thiết bị đầu cuối với vị trí và yêu cầu QoS

2. **Kết nối với Nodes** (Terminal Detail Card)
   - Mỗi terminal có thể kết nối với một node (satellite hoặc ground station)
   - Hệ thống sẽ kiểm tra khả năng kết nối dựa trên:
     - Khoảng cách
     - Tín hiệu
     - Băng thông khả dụng

3. **Truyền dữ liệu**
   - Khi connected, terminal có thể truyền dữ liệu
   - Hệ thống theo dõi metrics: latency, bandwidth, packet loss

4. **Visualization trên Map**
   - Terminals hiển thị trên Cesium map với màu sắc theo trạng thái
   - Đường kết nối (connection lines) giữa terminal và node

## 💡 Use Cases

### 1. **Testing & Simulation**
- Tạo nhiều terminals để test khả năng xử lý của mạng
- Mô phỏng các scenario khác nhau (thành phố đông đúc, vùng nông thôn)

### 2. **Resource Allocation Testing**
- Test thuật toán phân bổ tài nguyên (RL algorithms)
- Xem cách hệ thống xử lý khi có nhiều terminals cùng yêu cầu kết nối

### 3. **Network Planning**
- Phân tích mật độ terminals trong các khu vực
- Đánh giá nhu cầu băng thông và tài nguyên

### 4. **QoS Monitoring**
- Theo dõi chất lượng kết nối của từng terminal
- Phát hiện terminals có vấn đề về latency hoặc packet loss

## 🎮 Cách sử dụng

### Bước 1: Mở Terminal Controls
- Click nút **"📡 Terminals"** ở góc trên bên trái của map
- Panel điều khiển sẽ hiện ra

### Bước 2: Tạo Terminals
1. Nhập số lượng terminals (ví dụ: 20)
2. Click **"Generate Terminals"**
3. Đợi hệ thống tạo xong (có thể mất vài giây)

### Bước 3: Xem Terminals trên Map
- Terminals sẽ xuất hiện trên Cesium map
- Màu sắc:
  - **Xám**: Idle (chưa kết nối)
  - **Xanh lá**: Connected (đã kết nối)
  - **Vàng**: Transmitting (đang truyền dữ liệu)

### Bước 4: Kết nối Terminal với Node
1. Click vào một terminal trên map
2. Terminal Detail Card sẽ hiện ra
3. Click **"🔗 Connect"** để kết nối với node gần nhất
4. Xem connection metrics (latency, bandwidth, etc.)

### Bước 5: Theo dõi Status
- Xem thống kê trong Terminal Controls panel
- Theo dõi số lượng terminals ở các trạng thái khác nhau

## 📊 Thông tin hiển thị

### Terminal Detail Card hiển thị:
- **Position**: Vị trí (latitude, longitude, altitude)
- **Status**: Trạng thái hiện tại
- **QoS Requirements**: Yêu cầu chất lượng dịch vụ
  - Max Latency (ms)
  - Min Bandwidth (Mbps)
  - Max Loss Rate (%)
  - Priority
- **Connection Metrics** (nếu đã kết nối):
  - Latency (ms)
  - Bandwidth (Mbps)
  - Packet Loss Rate (%)
  - Signal Strength (dB)

## 🔄 Tích hợp với WebSocket

Terminal Controls tự động cập nhật real-time qua WebSocket:
- Khi terminal thay đổi trạng thái
- Khi có kết quả kết nối mới
- Khi metrics thay đổi

## ⚙️ Cấu hình

Có thể tùy chỉnh trong code:
- Số lượng terminals tối đa
- Vùng tạo terminals (bounds)
- Loại terminals (MOBILE, FIXED, VEHICLE, AIRCRAFT)
- QoS requirements mặc định

## 🎯 Tóm tắt

**Terminal Controls** là công cụ để:
1. ✅ **Tạo** terminals mới cho simulation
2. ✅ **Quản lý** terminals (xóa, theo dõi)
3. ✅ **Theo dõi** trạng thái và metrics
4. ✅ **Test** khả năng xử lý của mạng SAGIN
5. ✅ **Phân tích** nhu cầu tài nguyên và QoS

Đây là một phần quan trọng trong việc mô phỏng và test hệ thống SAGIN với nhiều thiết bị đầu cuối khác nhau.

