FROM python:3.9-slim

# Set working directory
WORKDIR /app/trading-strategy-backtester

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY src/requirements.txt ./src/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p input output logs cache

# Set environment variables
ENV PYTHONPATH=/app/trading-strategy-backtester
ENV BASE_DIR=/app/trading-strategy-backtester

# Default command
CMD ["python", "src/workflows/cli.py", "--help"]