# Sử dụng image OpenJDK 25
FROM openjdk:25-jdk

# Đặt thông tin tác giả
LABEL maintainer="anhnon0106@gmail.com"


# Tạo thư mục làm việc trong container
WORKDIR /app

# Sao chép file JAR từ máy chủ vào thư mục làm việc trong container
COPY target/core-0.0.1-SNAPSHOT.jar app.jar


# Chạy ứng dụng Spring Boot
ENTRYPOINT ["java", "-jar", "/app/app.jar"]


# Cấu hình để ứng dụng Spring Boot chạy trên cổng 8080 (HTTP)
EXPOSE 8080