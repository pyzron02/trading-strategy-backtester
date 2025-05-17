#!/bin/bash
set -e

# Create required directories if they don't exist
mkdir -p /app/frontend/temp
mkdir -p /app/frontend/output
mkdir -p /app/trading-strategy-backtester/input
mkdir -p /app/trading-strategy-backtester/output
mkdir -p /app/trading-strategy-backtester/logs
mkdir -p /app/trading-strategy-backtester/cache

# Set environment variables
export BACKTESTER_ROOT="/app/trading-strategy-backtester"
export PYTHONPATH="/app:/app/trading-strategy-backtester"
export BASE_DIR="/app/trading-strategy-backtester"

# Create config.json for frontend if it doesn't exist
if [ ! -f "/app/frontend/config.json" ]; then
    echo "Creating config.json from template..."
    cp /app/frontend/config.template.json /app/frontend/config.json
    
    # Update config with environment variables
    sed -i "s|/path/to/trading-strategy-backtester|$BACKTESTER_ROOT|g" /app/frontend/config.json
    sed -i "s|/path/to/trading-strategy-backtester/output|$BACKTESTER_ROOT/output|g" /app/frontend/config.json
    
    # Update temp directory path
    sed -i "s|/path/to/frontend/temp|/app/frontend/temp|g" /app/frontend/config.json
    
    # Update secret key if provided
    if [ -n "$SECRET_KEY" ]; then
        sed -i "s|REPLACE_WITH_YOUR_SECRET_KEY|$SECRET_KEY|g" /app/frontend/config.json
    else
        # Set a default secret key
        sed -i "s|REPLACE_WITH_YOUR_SECRET_KEY|trading_strategy_backtester_secret_key|g" /app/frontend/config.json
    fi
fi

echo "Starting Trading Strategy Backtester Frontend..."
cd /app/frontend
exec "$@"