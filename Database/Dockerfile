# 📍 database/Dockerfile
FROM mongo:6.0

# Install mongosh (MongoDB Shell) for health checks
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | apt-key add - && \
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list && \
    apt-get update && apt-get install -y mongodb-mongosh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
COPY ./mongo-init/ /docker-entrypoint-initdb.d/

# Set permissions for scripts
RUN if [ -d /docker-entrypoint-initdb.d ] && [ "$(ls -A /docker-entrypoint-initdb.d 2>/dev/null)" ]; then \
        chmod +x /docker-entrypoint-initdb.d/*.js 2>/dev/null || true; \
    fi

# Create directory for MongoDB data
RUN mkdir -p /data/db && chown -R mongodb:mongodb /data/db

# Expose MongoDB port
EXPOSE 27017

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD mongosh --eval "db.adminCommand('ping')"

# Start MongoDB
CMD ["mongod", "--bind_ip_all", "--auth"]