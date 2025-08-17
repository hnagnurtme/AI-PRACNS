# PBL4: Tối ưu hóa Phân bổ Tài nguyên bằng GenAI

## Tổng quan Dự án
Dự án tập trung vào **tối ưu hóa phân bổ tài nguyên** trong **Mạng Tích hợp Không gian–Hàng không–Mặt đất–Biển (SAGSINs)** bằng **Trí tuệ Nhân tạo Tạo sinh (GenAI)** và các thuật toán heuristic. Mục tiêu chính là giảm độ trễ truyền thông giữa các nút mạng (vệ tinh, UAV, trạm mặt đất/biển) đồng thời tối ưu hóa việc sử dụng băng thông, năng lượng và các tài nguyên khác.

---

## Kiến trúc Hệ thống

### 1. Client
- Các client gửi **yêu cầu (request)** đến người dùng ở các vị trí khác nhau.
- Client có thể là thiết bị cố định (ví dụ: trạm mặt đất) hoặc di động (ví dụ: UAV, tàu biển).
- Nhiệm vụ:
  - Gửi yêu cầu phân bổ tài nguyên đến server.
  - Nhận và xử lý phản hồi với kế hoạch phân bổ tài nguyên tối ưu.

### 2. Server Phân bổ Tài nguyên
- Nhận yêu cầu từ client.
- Xử lý yêu cầu bằng **thuật toán tối ưu** dựa trên:
  - Vị trí của client và người dùng.
  - Băng thông, độ trễ, và tải hiện tại của các nút SAGSIN (vệ tinh, UAV, trạm mặt đất/biển).
  - Thỏa thuận Cấp độ Dịch vụ (SLAs) hoặc mức độ ưu tiên của người dùng.
- Lưu trữ trạng thái mạng và lịch sử yêu cầu trong **MongoDB**.
- Giao tiếp với Server AI để lấy chiến lược phân bổ tối ưu.

### 3. Server AI/GenAI
- Thu thập dữ liệu lịch sử sử dụng mạng từ **MongoDB**.
- Huấn luyện và tạo ra **chiến lược phân bổ tài nguyên tối ưu** dựa trên điều kiện mạng và dự báo lưu lượng.
- Trả kết quả chiến lược tối ưu về Server Phân bổ Tài nguyên để áp dụng trong thời gian thực.

---

## Luồng Dữ liệu
```plaintext
Client A
   └─ Yêu cầu ─> Server Phân bổ Tài nguyên ─> Lấy thông tin nút SAGSIN (vệ tinh, UAV, trạm)
   └─ Lưu dữ liệu vào MongoDB ─> Server AI huấn luyện & tối ưu
   └─ Chiến lược tối ưu ─> Server Phân bổ Tài nguyên ─> Phản hồi ─> Client B
```

---

## Lưu trữ Dữ liệu (MongoDB)
Hệ thống sử dụng **MongoDB** để lưu trữ:
- **Nút SAGSIN**: Vị trí, băng thông, độ trễ, tải hiện tại.
  ```json
  {
    "nodeId": "string",
    "type": "string (vệ tinh/UAV/mặt đất/biển)",
    "location": { "lat": "float", "lon": "float" },
    "bandwidth": "float",
    "latency": "float",
    "load": "float",
    "timestamp": "ISODate"
  }
  ```
- **Yêu cầu**: Client, người nhận, thời gian, thời gian phản hồi, trạng thái.
  ```json
  {
    "requestId": "string",
    "clientId": "string",
    "recipientId": "string",
    "timestamp": "ISODate",
    "responseTime": "float",
    "status": "string (pending/success/failed)"
  }
  ```
- **Chiến lược AI**: Các kịch bản phân bổ tối ưu và kết quả mô phỏng.
  ```json
  {
    "strategyId": "string",
    "nodes": ["nodeId"],
    "allocationPlan": { "bandwidth": "float", "path": ["nodeId"] },
    "predictedLatency": "float",
    "timestamp": "ISODate"
  }
  ```

---

## Thuật toán Tối ưu hóa
- **Đầu vào**: Vị trí client và người dùng, trạng thái mạng SAGSIN.
- **Đầu ra**: Đường đi tối ưu và kế hoạch phân bổ tài nguyên.
- **Phương pháp**:
  - **Thuật toán Heuristic**: Tối ưu hóa Bầy đàn (PSO), Tối ưu hóa Đàn kiến (ACO).
  - **AI Tạo sinh**: Học Tăng cường (Reinforcement Learning) với Mô hình Tạo sinh để học và cải thiện kịch bản phân bổ.
- **Mục tiêu**:
  - Giảm độ trễ truyền thông.
  - Tối đa hóa thông lượng.
  - Cân bằng sử dụng tài nguyên (băng thông, năng lượng).

---

## Kết quả Mong đợi
- **GenAI** tạo ra nhiều phương án phân bổ tài nguyên tối ưu dựa trên điều kiện mạng.
- **Giảm độ trễ** trong truyền thông giữa các nút SAGSIN.
- **Cải thiện hiệu quả** sử dụng băng thông, năng lượng và tài nguyên mạng.
- **Khả năng thích ứng thời gian thực** thông qua mô hình dự đoán nhu cầu mạng.

---

## Công nghệ Sử dụng
- **Ngôn ngữ Lập trình**: Python (chính), có thể dùng Java/C++ cho một số thành phần.
- **Khung AI/ML**: PyTorch, TensorFlow, HuggingFace Transformers.
- **Thuật toán Tối ưu**: PSO, ACO, Học Tăng cường.
- **Cơ sở Dữ liệu**: MongoDB để lưu trữ trạng thái nút, yêu cầu và chiến lược AI.
- **Giao tiếp Client–Server**: REST API (qua FastAPI/Flask) hoặc gRPC cho hiệu suất cao.
- **Triển khai**: Docker để đóng container, Kubernetes để điều phối (tùy chọn).

---

## Cấu trúc Dự án
```plaintext
PBL4/
├── data/                 # Dữ liệu đầu vào, lịch sử yêu cầu, trạng thái nút
├── models/               # Mô hình AI/GenAI đã huấn luyện
├── src/                  # Mã nguồn cho client, server và thuật toán tối ưu
│   ├── client/           # Logic phía client
│   ├── server/           # Server Phân bổ Tài nguyên và Server AI
│   └── optimization/      # Thuật toán PSO, ACO, và RL
├── scripts/              # Script mô phỏng và tối ưu hóa
├── README_EN.md          # Tài liệu dự án bằng tiếng Anh
├── README_VN.md          # Tài liệu dự án bằng tiếng Việt
└── requirements.txt      # Các thư viện Python cần thiết
```

---

## Hướng dẫn Cài đặt
1. **Tải mã nguồn**:
   ```bash
   git clone https://github.com/your-repo/PBL4.git
   cd PBL4
   ```

2. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Thiết lập MongoDB**:
   - Cài đặt MongoDB cục bộ hoặc sử dụng dịch vụ đám mây (ví dụ: MongoDB Atlas).
   - Cấu hình chuỗi kết nối trong `src/server/config.py`.

4. **Chạy hệ thống**:
   - Khởi động Server Phân bổ Tài nguyên:
     ```bash
     python src/server/resource_server.py
     ```
   - Khởi động Server AI:
     ```bash
     python src/server/ai_server.py
     ```
   - Chạy mô phỏng client:
     ```bash
     python src/client/client.py
     ```

5. **Chạy mô phỏng**:
   - Sử dụng các script trong thư mục `scripts/` để mô phỏng điều kiện mạng và kiểm tra thuật toán tối ưu:
     ```bash
     python scripts/simulate_network.py
     ```

---

## Báo cáo Lỗi
Để báo cáo lỗi, sử dụng mẫu `bug.yml` có sẵn trong kho mã nguồn. Bao gồm:
- Tên hệ thống/dịch vụ (ví dụ: PBL4).
- Mô tả chi tiết về lỗi.
- Các bước tái hiện, hành vi mong đợi so với hành vi thực tế.
- Thông tin môi trường (hệ điều hành, phiên bản Python, v.v.).
- Log hoặc ảnh chụp màn hình để hỗ trợ gỡ lỗi.

Mẫu báo cáo lỗi: [bug.yml](bug.yml).

---

## Đóng góp
Chúng tôi hoan nghênh mọi đóng góp! Vui lòng làm theo các bước sau:
1. Fork kho mã nguồn.
2. Tạo nhánh tính năng (`git checkout -b feature/YourFeature`).
3. Commit thay đổi (`git commit -m "Add YourFeature"`).
4. Push nhánh (`git push origin feature/YourFeature`).
5. Tạo Pull Request.

---

## Giấy phép
Dự án được cấp phép theo Giấy phép MIT. Xem chi tiết trong file [LICENSE](LICENSE).