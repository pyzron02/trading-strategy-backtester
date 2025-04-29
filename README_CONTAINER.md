# Containerization Guide

This document describes how the trading-strategy-backtester project has been updated to support containerization alongside a frontend application.

## Path Management

The project now uses a centralized path management system to avoid absolute paths and make the code more portable:

- The `src/utils/path_manager.py` module provides a unified way to handle paths
- All paths are now relative to a configurable base directory
- This makes it possible to run the code in different environments, including containers

## Directory Structure

The recommended directory structure when using this backend with a frontend is:

```
/project-root
  /trading-strategy-backtester     # This backend project
    /src
    /input
    /output
    /logs
    /cache
    ...
  /frontend                       # Frontend project
    ...
```

Inside the Docker environment, these will be mapped to:

```
/app
  /trading-strategy-backtester     # This backend project
  /frontend                       # Frontend project
```

## Using the Path Manager

The `PathManager` class provides several useful methods:

- `get_path(name)`: Get a specific directory path (base_dir, src_dir, etc.)
- `join_path(base, *paths)`: Join paths relative to a base directory
- `ensure_dir(path)`: Create a directory if it doesn't exist
- `ensure_base_dirs()`: Create all standard directories
- `rel_path(path)`: Convert a path to be relative to base_dir
- `abs_path(rel_path)`: Convert a relative path to absolute

## Modified Components

The following components have been updated to use the path manager:

1. Directory management:
   - `src/utils/ensure_directories.py`
   - `src/engine/setup_directories.py`

2. Logging system:
   - `src/engine/logging_system.py`

3. Data handling:
   - `src/data_preprocessing/data_setup.py`

4. Workflow utilities:
   - `src/workflows/workflow_utils.py`

## Integration with Frontend

### Setting Up the Project Structure

1. Place the frontend and backend projects side by side:

```bash
mkdir my-trading-app
cd my-trading-app
git clone <trading-strategy-backtester-repo> trading-strategy-backtester
git clone <frontend-repo> frontend
```

2. Modify the docker-compose.yml to include both projects:

```yaml
# Use the provided docker-compose.yml in the trading-strategy-backtester directory
# or create a new one in the parent directory
```

### Frontend-Backend Communication

The frontend can communicate with the backend through:

1. **REST API**: If you add a REST API to the backend, the frontend can call it.
2. **Shared Volumes**: The frontend can read output files directly from the mounted volumes.
3. **WebSocket**: For real-time updates, you can add a WebSocket server to the backend.

### Sharing Data Between Frontend and Backend

To share data between frontend and backend containers:

1. Use a shared volume that both containers can access:

```yaml
volumes:
  shared-data:

services:
  backtester:
    # ...other config...
    volumes:
      - shared-data:/app/trading-strategy-backtester/output
  
  frontend:
    # ...other config...
    volumes:
      - shared-data:/app/frontend/public/data
```

2. Configure the frontend to read from this shared data directory.

## Docker Setup

### Docker Integration

The provided Dockerfile already sets up the backend environment:

1. It uses the BASE_DIR environment variable:
   ```
   ENV BASE_DIR=/app/trading-strategy-backtester
   ```

2. It properly mounts volumes for persistent data:
   ```
   volumes:
     - ./input:/app/trading-strategy-backtester/input
     - ./output:/app/trading-strategy-backtester/output
     - ./logs:/app/trading-strategy-backtester/logs
     - ./cache:/app/trading-strategy-backtester/cache
   ```

### Docker Compose for Full-Stack Setup

The following docker-compose.yml shows how to integrate both frontend and backend:

```yaml
version: '3'

services:
  backtester:
    build:
      context: ./trading-strategy-backtester
      dockerfile: Dockerfile
    volumes:
      - ./trading-strategy-backtester/input:/app/trading-strategy-backtester/input
      - ./trading-strategy-backtester/output:/app/trading-strategy-backtester/output
      - ./trading-strategy-backtester/logs:/app/trading-strategy-backtester/logs
      - ./trading-strategy-backtester/cache:/app/trading-strategy-backtester/cache
      - shared-data:/app/trading-strategy-backtester/output/shared
    environment:
      - BASE_DIR=/app/trading-strategy-backtester
    networks:
      - app-network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - shared-data:/app/frontend/public/data
    depends_on:
      - backtester
    networks:
      - app-network

volumes:
  shared-data:

networks:
  app-network:
    driver: bridge
```

## Using the Containerized Application

### Starting the Application

1. From the project root directory (containing both subdirectories):

```bash
cd trading-strategy-backtester
docker-compose up
```

2. To run in detached mode:

```bash
docker-compose up -d
```

### Running Backtests

1. Through the CLI within the container:

```bash
docker-compose exec backtester python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL
```

2. By modifying the docker-compose.yml command:

```yaml
services:
  backtester:
    # ...other config...
    command: python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL
```

### Accessing Output Files

1. In the host machine:
   - All output files will be available in the `./trading-strategy-backtester/output` directory

2. From the frontend container:
   - Output files will be available at `/app/frontend/public/data`

## Development Workflow

1. **Local Development**: 
   - Work on the backend code directly in the `trading-strategy-backtester` directory
   - Work on the frontend code directly in the `frontend` directory

2. **Testing the Integration**:
   - Run `docker-compose up` to start both services
   - Make changes to either codebase
   - Restart containers to see changes: `docker-compose restart`

3. **Rebuilding After Changes**:
   - If you change Dockerfile or requirements: `docker-compose build`
   - Then restart: `docker-compose up`

## Production Deployment

For production deployment:

1. Create a production-specific docker-compose file (docker-compose.prod.yml)
2. Set appropriate environment variables for production
3. Configure proper networking and security settings
4. Use a reverse proxy like Nginx to handle incoming requests

Example production docker-compose:

```yaml
version: '3'

services:
  backtester:
    build:
      context: ./trading-strategy-backtester
      dockerfile: Dockerfile.prod
    restart: always
    # other production settings...

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    restart: always
    # other production settings...

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx:/etc/nginx/conf.d
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    depends_on:
      - frontend
      - backtester
```