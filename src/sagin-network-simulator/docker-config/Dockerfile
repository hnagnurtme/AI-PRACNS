# File: src/sagin-network-simulator/docker-config/Dockerfile

# === GIAI ĐOẠN 1: BUILD (Giữ nguyên) ===
FROM maven:3.9.5-eclipse-temurin-17 AS build

WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline

COPY src ./src

# Đóng gói ứng dụng thành file JAR
RUN mvn package -DskipTests

# === GIAI ĐOẠN 2: RUNTIME ===
FROM eclipse-temurin:17-jre-alpine

# LƯU Ý: ĐÃ XÓA JAVA_TOOL_OPTIONS KHỎI ĐÂY

# Tên file JAR đã được SHADE 
ARG UBER_JAR_FILE=target/sagin-network-simulator-1.0.0-SNAPSHOT-uber.jar

WORKDIR /app

# Sao chép file JAR đã build và đổi tên thành app.jar
COPY --from=build /app/${UBER_JAR_FILE} /app/app.jar

# SỬ DỤNG CMD: Để cho phép command trong compose ghi đè
CMD ["java", "-jar", "app.jar"]