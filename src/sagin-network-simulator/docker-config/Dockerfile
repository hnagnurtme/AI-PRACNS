# Build stage vẫn dùng Maven multiarch
FROM maven:3.9.5-eclipse-temurin-17 AS build

WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline

COPY src ./src
RUN mvn package -DskipTests

# Runtime stage: dùng multiarch image ARM
FROM eclipse-temurin:17-jre-jammy

WORKDIR /app
ARG UBER_JAR_FILE=target/sagin-network-simulator-1.0.0-SNAPSHOT-uber.jar
COPY --from=build /app/${UBER_JAR_FILE} /app/app.jar

CMD ["java", "-jar", "app.jar"]
