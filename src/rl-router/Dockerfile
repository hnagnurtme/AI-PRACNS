# Sử dụng mirror châu Á để tránh lỗi TLS handshake timeout
# (daocloud.io có tốc độ tốt tại Việt Nam)
FROM docker.m.daocloud.io/library/python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cài các thư viện hệ thống cần thiết (nhỏ gọn)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Tạo user không phải root để chạy an toàn
RUN groupadd -r app && useradd -r -g app -m app

WORKDIR /app

# Copy requirements trước để tối ưu cache build
COPY requirements.txt /app/requirements.txt

# Cài pip và dependency
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy toàn bộ mã nguồn
COPY . /app

# Đảm bảo quyền sở hữu file đúng user
RUN chown -R app:app /app

USER app

# Expose port mà TCP service sẽ sử dụng
EXPOSE 65000

# Chạy service chính
CMD ["python", "-u", "-m", "python.service.tcp_inference_service"]
