# Multi-stage build
FROM maven:3.9.6-eclipse-temurin-21 AS build

WORKDIR /app
COPY pom.xml .
RUN mvn dependency:resolve
COPY src ./src
RUN mvn clean package -DskipTests
RUN ls -la target/

# Runtime stage
FROM eclipse-temurin:21-jdk

# Install dependencies including noVNC
RUN apt-get update && apt-get install -y \
    libgtk-3-0 libxtst6 libxrender1 libxi6 \
    libfontconfig1 libgl1-mesa-dri libgl1 \
    libasound2t64 libfreetype6 x11-apps \
    xvfb x11vnc fluxbox wget unzip git \
    python3 python3-numpy supervisor \
    xterm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy JAR
COPY --from=build /app/target/*.jar ./
RUN if [ -f ChatApp-*-shaded.jar ]; then mv ChatApp-*-shaded.jar app.jar; else mv ChatApp-*.jar app.jar; fi

# Download JavaFX
RUN wget -q https://download2.gluonhq.com/openjfx/21.0.2/openjfx-21.0.2_linux-x64_bin-sdk.zip -O javafx.zip \
    && unzip -q javafx.zip && mv javafx-sdk-21.0.2 javafx && rm javafx.zip

# Install noVNC
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc \
    && git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify

# Create chat history directory
RUN mkdir -p /app/chat_history

# Create debug script
RUN echo '#!/bin/bash' > debug.sh && \
    echo 'echo "=== CHATAPP DEBUG ==="' >> debug.sh && \
    echo 'echo "Java version:"' >> debug.sh && \
    echo 'java -version' >> debug.sh && \
    echo 'echo "JAR info:"' >> debug.sh && \
    echo 'ls -la app.jar' >> debug.sh && \
    echo 'echo "JavaFX modules:"' >> debug.sh && \
    echo 'java --list-modules | grep javafx' >> debug.sh && \
    echo 'echo "Display info:"' >> debug.sh && \
    echo 'echo "DISPLAY=$DISPLAY"' >> debug.sh && \
    echo 'xdpyinfo -display :99 | head -5' >> debug.sh && \
    echo 'echo "Starting app..."' >> debug.sh && \
    echo 'java --module-path javafx/lib --add-modules javafx.controls,javafx.fxml -Djava.awt.headless=false -Dprism.verbose=true -jar app.jar' >> debug.sh && \
    chmod +x debug.sh

# Create supervisord config
RUN echo '[supervisord]' > /etc/supervisor/conf.d/supervisord.conf && \
    echo 'nodaemon=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=root' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:xvfb]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=Xvfb :99 -screen 0 1600x1000x24 -ac' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/xvfb.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/xvfb.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:fluxbox]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=fluxbox' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'environment=DISPLAY=:99' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/fluxbox.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/fluxbox.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:x11vnc]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=x11vnc -display :99 -nopw -listen 0.0.0.0 -forever -shared' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/x11vnc.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/x11vnc.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:novnc]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=/opt/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/novnc.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/novnc.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:chatapp]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=java --module-path javafx/lib --add-modules javafx.controls,javafx.fxml -Djava.awt.headless=false -Dprism.verbose=true -jar app.jar' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'directory=/app' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'environment=DISPLAY=:99' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=false' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'startsecs=10' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/chatapp.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/chatapp.log' >> /etc/supervisor/conf.d/supervisord.conf

# Create startup script
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "🚀 Starting ChatApp with integrated noVNC..."' >> start.sh && \
    echo 'echo "📱 Web UI will be available at http://localhost:6080"' >> start.sh && \
    echo 'echo "💾 Chat files will be saved in: /app/chat_history"' >> start.sh && \
    echo 'supervisord -c /etc/supervisor/conf.d/supervisord.conf &' >> start.sh && \
    echo 'echo "⏳ Waiting 10 seconds for services to start..."' >> start.sh && \
    echo 'sleep 10' >> start.sh && \
    echo 'echo "🔍 Running debug..."' >> start.sh && \
    echo './debug.sh' >> start.sh && \
    echo 'echo "✅ Debug completed. Check logs with: docker exec <container> cat /var/log/chatapp.log"' >> start.sh && \
    echo 'wait' >> start.sh && \
    chmod +x start.sh

# Create conditional startup script
RUN echo '#!/bin/bash' > conditional-start.sh && \
    echo 'if [ "$SERVER_ONLY" = "true" ]; then' >> conditional-start.sh && \
    echo '  echo "🖥️ Starting as Socket Server only..."' >> conditional-start.sh && \
    echo '  java -cp app.jar com.chatapp.chatapp.model.services.SocketChatServer' >> conditional-start.sh && \
    echo 'else' >> conditional-start.sh && \
    echo '  echo "📱 Starting as Chat Client..."' >> conditional-start.sh && \
    echo '  exec ./start.sh' >> conditional-start.sh && \
    echo 'fi' >> conditional-start.sh && \
    chmod +x conditional-start.sh

# Expose both VNC and HTTP ports
EXPOSE 5900 6080 8888

ENTRYPOINT ["./conditional-start.sh"]