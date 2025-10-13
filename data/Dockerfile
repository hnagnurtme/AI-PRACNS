# --- Dùng image Python nhẹ ---
FROM python:3.11-slim

# --- Thiết lập thư mục làm việc trong container ---
WORKDIR /app

# --- Sao chép file cấu hình và dependencies ---
COPY requirements.txt .

# --- Cài đặt dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Sao chép toàn bộ code vào container ---
COPY . .

# --- Thiết lập biến môi trường (nếu cần override từ docker-compose) ---
ENV MONGO_URL="mongodb://user:password123@mongodb:27017/"

# --- Lệnh khởi động chính ---

CMD ["python", "schedule_node_update.py"]
