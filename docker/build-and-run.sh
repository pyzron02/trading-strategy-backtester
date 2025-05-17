#!/bin/bash
set -e

echo "Building and starting the Trading Strategy Backtester container..."

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if the required directories exist
if [ ! -d "$PROJECT_ROOT/src" ]; then
    echo "Error: src directory not found at $PROJECT_ROOT/src"
    echo "Please make sure the backtester code is available at the expected location"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/frontend" ]; then
    echo "Error: frontend directory not found at $PROJECT_ROOT/frontend"
    echo "Please make sure the frontend code is available at the expected location"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p "$PROJECT_ROOT/input"
mkdir -p "$PROJECT_ROOT/output"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/cache"
mkdir -p "$PROJECT_ROOT/frontend/temp"
mkdir -p "$PROJECT_ROOT/frontend/output"

# Make the entrypoint script executable
chmod +x "$SCRIPT_DIR/docker-entrypoint.sh"

# Build and start the container
cd "$SCRIPT_DIR" && docker-compose up --build

echo "Container is now running. Access the web interface at http://localhost:5000"