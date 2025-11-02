
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies required by some Python packages (torch, numpy, etc.)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   ca-certificates \
	   libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the service
RUN groupadd -r app && useradd -r -g app -m app

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
	&& python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure the app user owns the application files
RUN chown -R app:app /app

USER app

# Expose port used by tcp_inference_service.py
EXPOSE 65000

# Default command: run the TCP inference service as a module to support relative imports
CMD ["python", "-u", "-m", "python.service.tcp_inference_service"]

