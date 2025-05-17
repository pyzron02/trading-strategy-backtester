# Trading Strategy Backtester Docker Container

This setup allows you to run both the trading strategy backtester engine and the frontend web interface in a single Docker container.

## Directory Structure

The container is designed for this updated directory structure:

```
trading-strategy-backtester/       # Root project directory
├── src/                           # The backtester engine code
├── frontend/                      # The frontend web interface code
├── docker/                        # Docker configuration files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-entrypoint.sh
│   └── README.md
├── input/                         # Input data directory
├── output/                        # Output data directory
├── logs/                          # Log files
└── cache/                         # Cache files
```

## Building and Running

1. Make sure you have Docker and Docker Compose installed on your system.

2. Build and start the container:

```bash
cd trading-strategy-backtester
./docker/build-and-run.sh
```

Alternatively, you can use Docker Compose directly:

```bash
cd trading-strategy-backtester
docker-compose -f docker/docker-compose.yml up --build
```

3. Access the web interface at http://localhost:5000

## Environment Variables

You can customize the container behavior with these environment variables:

- `SECRET_KEY`: Secret key for the Flask application (default: trading_strategy_backtester_secret_key)
- `BACKTESTER_ROOT`: Path to the backtester inside the container (default: /app/trading-strategy-backtester)

## Volume Mounts

The docker-compose.yml file maps several directories between your host and the container:

- `../input` → `/app/trading-strategy-backtester/input`
- `../output` → `/app/trading-strategy-backtester/output`
- `../logs` → `/app/trading-strategy-backtester/logs`
- `../cache` → `/app/trading-strategy-backtester/cache`
- `../frontend/temp` → `/app/frontend/temp`
- `../frontend/output` → `/app/frontend/output`
- `../frontend/app.py` → `/app/frontend/app.py`

This allows your backtest results to persist across container restarts.

## Custom Configuration

The container automatically creates a `config.json` file for the frontend from the template if it doesn't exist.

If you need to customize the configuration, you can either:

1. Edit the `config.template.json` file in the frontend directory before building the container
2. Mount a custom config.json file directly into the container
3. Set appropriate environment variables when starting the container

## Troubleshooting

If you encounter any issues:

1. Check the container logs:
```bash
docker-compose logs
```

2. Make sure the directory structure matches what's expected
3. Verify that both projects have the required files and dependencies
4. Ensure the proper permissions are set for the mounted volumes