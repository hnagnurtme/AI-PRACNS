FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools for PyTorch Geometric
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 5000

# Set Python path to include Reinforcement folder (will be mounted)
ENV PYTHONPATH="${PYTHONPATH}:/app/Reinforcement"

# Run the application
CMD ["python", "app.py"]

