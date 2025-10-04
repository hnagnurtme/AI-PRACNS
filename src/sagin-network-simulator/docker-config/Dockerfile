# File: src/sagin-network-simulator/docker-config/Dockerfile

# === GIAI ĐOẠN 1: BUILD ===
FROM maven:3.9.5-eclipse-temurin-17 AS build

WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline

COPY src ./src

# Đóng gói ứng dụng thành file JAR
RUN mvn package -DskipTests

# === GIAI ĐOẠN 2: RUNTIME ===
FROM eclipse-temurin:17-jre-alpine

# Tên file JAR được tạo ra
ARG JAR_FILE=target/sagin-network-simulator-1.0.0-SNAPSHOT.jar

WORKDIR /app

# Sao chép file JAR đã build từ builder stage
COPY --from=build /app/${JAR_FILE} /app/app.jar

# CMD thay vì ENTRYPOINT để cho phép docker-compose override hoàn toàn
CMD ["java", "-jar", "app.jar"]