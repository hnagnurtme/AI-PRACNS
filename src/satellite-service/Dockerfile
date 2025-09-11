# -----------------------------
# Stage 1: Build
# -----------------------------
FROM openjdk:17-slim AS build
    
# Install Maven
RUN apt-get update && apt-get install -y maven && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc trong container
WORKDIR /app

# Copy pom.xml và tải dependencies trước (cache hiệu quả)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy toàn bộ source code
COPY src ./src

# Build project, tạo jar trong target/
RUN mvn clean package -DskipTests

# -----------------------------
# Stage 2: Run
# -----------------------------
FROM openjdk:17-slim

# Thư mục chạy ứng dụng
WORKDIR /app

# Copy JAR từ stage build
COPY --from=build /app/target/satellite-service-0.1.0.jar ./satellite-service.jar

# Mở port service
EXPOSE 6000

# Chạy ứng dụng
ENTRYPOINT ["java", "-jar", "satellite-service.jar"]
