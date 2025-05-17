FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=production
ENV PYTHONPATH=/app:/app/trading-strategy-backtester
ENV BASE_DIR=/app/trading-strategy-backtester
ENV BACKTESTER_ROOT=/app/trading-strategy-backtester

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Copy backtester requirements first
COPY requirements.txt /app/trading-strategy-backtester/
COPY src/requirements.txt /app/trading-strategy-backtester/src/

# Copy frontend requirements
COPY frontend/requirements.txt /app/frontend/

# Install dependencies
RUN pip install --no-cache-dir -r /app/trading-strategy-backtester/requirements.txt
RUN pip install --no-cache-dir -r /app/trading-strategy-backtester/src/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Create necessary directories for backtester
RUN mkdir -p /app/trading-strategy-backtester/input
RUN mkdir -p /app/trading-strategy-backtester/output
RUN mkdir -p /app/trading-strategy-backtester/logs
RUN mkdir -p /app/trading-strategy-backtester/cache

# Create necessary directories for frontend
RUN mkdir -p /app/frontend/temp
RUN mkdir -p /app/frontend/output

# Copy backtester files (excluding frontend and docker directories)
COPY AWS_DEPLOYMENT.md LICENSE README.md README_CONTAINER.md docker-compose.parent.yml /app/trading-strategy-backtester/
COPY src/ /app/trading-strategy-backtester/src/
COPY input/ /app/trading-strategy-backtester/input/
COPY tests/ /app/trading-strategy-backtester/tests/
COPY test_logs/ /app/trading-strategy-backtester/test_logs/

# Copy frontend files
COPY frontend /app/frontend/

# Create symbolic link for frontend app to access backtester
RUN ln -s /app/trading-strategy-backtester /trading-strategy-backtester

# Copy entrypoint script and make it executable
COPY docker/docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Expose port
EXPOSE 5000

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the application
CMD ["python", "/app/frontend/app.py"]