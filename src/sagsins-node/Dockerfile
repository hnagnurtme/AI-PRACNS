# -----------------------------------------------------------
# Build stage: Maven multiarch
# -----------------------------------------------------------
FROM maven:3.9.5-eclipse-temurin-17 AS build

WORKDIR /app

# Tải dependencies
COPY pom.xml .
RUN mvn dependency:go-offline

# Build ứng dụng
COPY src ./src
RUN mvn package -DskipTests

# -----------------------------------------------------------
# Runtime stage: dùng ARM hoặc AMD đều được (multiarch)
# -----------------------------------------------------------

FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

# SỬA LỖI: Cập nhật tên file JAR để khớp với finalName trong pom.xml
# Tên file thực tế: saginsimulator-1.0.0-SNAPSHOT-uber.jar
ARG UBER_JAR_FILE=target/saginsimulator-1.0.0-SNAPSHOT-uber.jar 

# Copy JAR từ Build Stage
COPY --from=build /app/${UBER_JAR_FILE} /app/app.jar

ENTRYPOINT ["java", "-jar", "app.jar"]