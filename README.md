# Trading Strategy Backtester

A robust framework for backtesting, optimizing, and evaluating trading strategies using historical market data, with a focus on Monte Carlo simulations for strategy validation.

## System Architecture

The trading strategy backtester is built with a modular architecture designed for flexibility, extensibility, and performance. Here's how the components work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Interface                                   │
│                                                                             │
│  ┌─────────────┐        ┌─────────────┐       ┌──────────────┐             │
│  │   Simple    │        │ Optimization │       │   Monte      │             │
│  │  Workflow   │        │   Workflow   │       │   Carlo      │             │
│  └─────────────┘        └─────────────┘       └──────────────┘             │
│         │                      │                      │                     │
│         └──────────────────────┼──────────────────────┘                     │
│                                │                                            │
│                   ┌────────────▼────────────┐                               │
│                   │    Unified Workflow     │                               │
│                   └────────────┬────────────┘                               │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                            Core Engine                                      │
│                                                                             │
│  ┌────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │   Data     │    │ Backtesting │    │ Performance │    │  Results     │  │
│  │ Management ├───▶│   Engine    ├───▶│ Evaluation  ├───▶│ Management   │  │
│  └────────────┘    └─────────────┘    └─────────────┘    └──────────────┘  │
│                           │                                      ▲          │
│                           ▼                                      │          │
│                    ┌─────────────┐                      ┌────────┴─────┐   │
│                    │  Strategy   │                      │              │   │
│                    │  Execution  │                      │  Reporting   │   │
│                    └─────────────┘                      │              │   │
│                           │                             └──────────────┘   │
└───────────────────────────┼─────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                        Strategy Components                                  │
│                                                                             │
│  ┌─────────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │  Strategy   │    │ Built-in   │    │   Custom      │    │  Strategy    │ │
│  │  Registry   ├───▶│ Strategies ├───▶│  Strategies   ├───▶│  Parameters  │ │
│  └─────────────┘    └────────────┘    └───────────────┘    └──────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Historical market data → Data Management → Backtest Engine
2. **Processing**: Strategy signals → Trade execution → Portfolio updates
3. **Output**: Performance metrics → Results → Visualization

### Components Interaction

- **Workflows**: Orchestrate multiple steps and provide a high-level API
- **Engine**: Handles core backtesting functionality
- **Strategies**: Contain trading logic that generates signals
- **Evaluators**: Calculate performance metrics and visualize results

## Features

- **Monte Carlo Testing:** Validate strategy robustness through multiple simulations
  - Data permutation techniques for statistical validation
  - P-value calculations for key performance metrics
  - Advanced equity curve and trade distribution visualization
  
- **Backtesting Engine:** Event-driven backtesting system
  - Customizable strategy implementation
  - Detailed performance metrics and trade logging
  - Portfolio value tracking and equity curve generation

- **Strategy Support:** 
  - Built-in strategies (SimpleStock, MACrossover, AuctionMarket, MultiPosition)
  - Strategy registry for custom implementations
  - Parameter optimization

- **Performance Metrics:** 
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Trade statistics

- **Parallel Processing:** 
  - Multi-core support for faster simulations
  - Configurable worker count
  - Parallel execution of backtests

## Directory Structure

The repository is organized with a new structure:

```
trading-strategy-backtester/
├── src/                    # Core backtesting engine components
│   ├── engine/                 # Core backtesting engine components
│   │   ├── run_backtest.py         # Main backtesting engine
│   │   ├── evaluate_performance.py # Performance metric calculations
│   │   ├── data_management.py      # Data loading and preprocessing
│   │   ├── parameter_management.py # Strategy parameter handling
│   │   ├── results_management.py   # Output and results handling
│   │   ├── parallel_testing.py     # Parallel execution support
│   │   ├── smart_cache.py          # Caching system for performance
│   │   └── logging_system.py       # Logging infrastructure
│   │
│   ├── data_preprocessing/     # Data preparation tools
│   ├── evaluators/             # Performance evaluation utilities
│   ├── monte_carlo/            # Monte Carlo simulation implementation
│   ├── optimizers/             # Parameter optimization components
│   ├── strategies/             # Trading strategy implementations
│   └── workflows/              # High-level workflow orchestrators
│
├── frontend/               # Web interface for the backtester
│   ├── app.py                  # Flask application for web interface
│   ├── static/                 # Static assets (CSS, JS)
│   ├── templates/              # HTML templates for the UI
│   ├── requirements.txt        # Frontend dependencies
│   └── config.json             # Frontend configuration
│
├── docker/                 # Docker configuration
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── docker-entrypoint.sh    # Container entrypoint script
│   └── build-and-run.sh        # Script to build and run the container
│
├── input/                  # Stock data input files
│   ├── stock_data.csv          # Historical price data
│   ├── parameters/             # Strategy parameter files
│   ├── parameter_grids/        # Parameter optimization grid definitions
│   └── workflow_configs/       # Workflow configuration files
│
├── output/                 # Test results and visualizations
├── logs/                   # Log files
├── cache/                  # Cache files for faster processing
├── tests/                  # Automated tests
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT license
```

## Installation

### Method 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-strategy-backtester
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set up data:**
   ```bash
   python src/data_preprocessing/data_setup.py --tickers AAPL,MSFT,GOOG,NVDA
   ```

### Method 2: Docker Installation

The project provides a comprehensive Docker setup that handles all dependencies and configuration automatically. This is the recommended method for most users as it simplifies installation and ensures consistency across different environments.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 19.03 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.27 or later)

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-strategy-backtester
   ```

2. **Build and run the integrated container:**
   ```bash
   # Using the convenience script (recommended)
   chmod +x docker/build-and-run.sh
   ./docker/build-and-run.sh
   ```

   The script automatically:
   - Creates necessary directories (input, output, logs, cache)
   - Sets proper permissions
   - Builds and starts the Docker container

3. **Alternatively, use Docker Compose directly:**
   ```bash
   # First ensure required directories exist
   mkdir -p input output logs cache frontend/temp frontend/output
   
   # Then build and start the container
   docker-compose -f docker/docker-compose.yml up --build
   ```

4. **Access the web interface:**
   Open your browser to http://localhost:5000 to use the integrated frontend interface.

#### Docker Configuration Details

The Docker setup provides:

- **Volume Mapping:** Your local directories (input, output, logs, cache) are mapped to the container
- **Live Development:** Frontend files can be edited on your local machine and changes reflect instantly
- **Automatic Configuration:** The container generates necessary config files on first run

#### Environment Variables

You can customize the Docker setup by setting these environment variables:

- `SECRET_KEY`: Custom secret key for the Flask application
- `BACKTESTER_ROOT`: Custom path to the backtester inside the container

Example with custom environment variables:
```bash
SECRET_KEY=my_custom_secret_key docker-compose -f docker/docker-compose.yml up
```

#### Container Management

Common operations:

- **View logs:** `docker-compose -f docker/docker-compose.yml logs`
- **Stop container:** `docker-compose -f docker/docker-compose.yml down`
- **Restart container:** `docker-compose -f docker/docker-compose.yml restart`
- **Remove container and rebuild:** `docker-compose -f docker/docker-compose.yml down --rmi all`

## Usage

### Configuration Files

The most flexible way to use the backtester is with configuration files. The system supports both static and dynamically generated configurations:

#### Dynamically Generated Configurations

When using the web interface, configurations are automatically generated based on your selections:

1. Configure your strategy, parameters, and workflow options in the web interface
2. The system generates a temporary JSON configuration file when you run the backtest
3. After execution, results are stored in the output directory while the temporary config is cleaned up

This approach simplifies the process of creating and managing configuration files, especially for new users.

#### Custom Configuration Files

You can also create your own custom configuration files and run them directly:

```bash
python src/workflows/cli.py --config /path/to/your/config.json
```

Or directly with the unified workflow runner:

```bash
python src/workflows/unified_workflow.py /path/to/your/config.json
```

#### Configuration Structure

Workflow configuration files use a standardized JSON format:

```json
{
  "workflow_type": "simple",           // Type of workflow to run
  "common_params": {                   // Common parameters for all strategies
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "data_dir": "input",
    "tickers": ["SPY", "AAPL"],
    "initial_capital": 100000.0,
    "commission": 0.001,
    "plot": true,
    "verbose": false
  },
  "strategies": {                      // Strategies to run
    "MACrossover": {                   // Strategy name
      "param_file": "input/parameters/MACrossover_params.json",   // Optional: path to parameter file
      "grid_file": "input/parameter_grids/MACrossover_grid.json", // Optional: path to grid file
      "parameters": {                  // Optional: inline parameters (if param_file not provided)
        "fast_period": 10,
        "slow_period": 50,
        "signal_period": 9
      },
      "parameter_grid": {              // Optional: inline parameter grid (if grid_file not provided)
        "fast_period": [5, 10, 15, 20],
        "slow_period": [30, 50, 100, 200],
        "signal_period": [5, 9, 14]
      },
      "optimization": {                // Optional: optimization parameters
        "n_trials": 50,
        "optimization_metric": "sharpe_ratio",
        "max_combinations": 100
      },
      "monte_carlo": {                 // Optional: Monte Carlo parameters
        "n_simulations": 100,
        "enhanced_plots": true
      },
      "walkforward": {                 // Optional: Walk-forward parameters
        "window_size": 252,
        "step_size": 63
      }
    }
  }
}
```

### CLI Interface

You can also use the backtester through the CLI interface with command-line arguments:

```bash
python src/workflows/cli.py --workflow [simple|optimization|monte_carlo|complete] \
    --strategy [StrategyName] \
    --tickers [Ticker1,Ticker2,...] \
    --start-date YYYY-MM-DD \
    --end-date YYYY-MM-DD \
    [--additional-options]
```

### Simple Backtest

To run a simple backtest with a specific strategy:

```bash
python src/utils/run_simple_workflow.py \
    --strategy MACrossover \
    --tickers AAPL,MSFT \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --verbose
```

### Monte Carlo Simulation

For robust strategy validation using Monte Carlo simulations:

```bash
python run_trade_monte_carlo.py \
    --strategy MultiPosition \
    --tickers NVDA,GOOG \
    --num-simulations 100 \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --enhanced-plots \
    --verbose
```

### Optimization Workflow

To optimize strategy parameters:

```bash
python src/workflows/cli.py --workflow optimization \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --n-trials 50 \
    --verbose
```

### Complete Workflow

Run the entire workflow including optimization, backtesting, and Monte Carlo simulation:

```bash
python src/workflows/cli.py --workflow complete \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --n-trials 20 \
    --n-simulations 100 \
    --enhanced-plots \
    --verbose
```

### Docker Usage

When using Docker, you can run commands inside the container:

```bash
docker-compose -f docker/docker-compose.yml exec trading-backtester python src/workflows/cli.py --workflow simple \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01
```

Or configure the command in docker-compose.yml:

```yaml
services:
  trading-backtester:
    # ... other settings ...
    command: python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL
```

### Web Interface

The integrated web interface provides a user-friendly way to configure and run backtests without writing any code or configuration files:

#### Accessing the Web Interface

1. After starting the Docker container, open your browser to http://localhost:5000
2. The landing page presents a form to configure your backtest

#### Main Features

- **Strategy Selection**: Choose from available built-in strategies
- **Parameter Configuration**: Configure strategy parameters with appropriate ranges
- **Multiple Workflow Types**: Run simple backtests, optimizations, Monte Carlo simulations, or complete workflows
- **Visualization**: View results with interactive charts and performance metrics
- **Results Management**: Browse and compare results from previous runs

#### Creating and Running a Backtest

1. Select a strategy from the dropdown menu
2. Enter tickers, date range, and initial capital
3. Configure strategy-specific parameters
4. Choose a workflow type (simple, optimization, monte_carlo, walkforward, complete)
5. Click "Run Backtest" to execute
6. View results in the Results section once processing is complete

## Monte Carlo Simulation Process

The Monte Carlo simulation framework creates synthetic but statistically similar market conditions to test how a trading strategy performs under different scenarios. This helps determine if a strategy's performance is due to actual edge or random chance.

### Overall Process Flow

```
┌────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌──────────────┐
│ Load Data  │────▶│ Run Original │────▶│ Generate Data │────▶│ Run Monte   │────▶│ Analyze      │
│            │     │ Backtest     │     │ Permutations  │     │ Carlo Tests │     │ Results      │
└────────────┘     └──────────────┘     └───────────────┘     └─────────────┘     └──────────────┘
```

### Data Permutation Methods

The framework implements multiple permutation methods to create synthetic market data:

1. **Returns Permutation**: Shuffles daily returns while preserving their distribution
2. **Block Permutation**: Preserves short-term autocorrelation by shuffling blocks of data
3. **Stationary Bootstrap**: Uses variable-length blocks for more realistic simulations

### Statistical Analysis

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Original       │     │ Calculate       │     │ Compare with  │     │ Generate        │
│ Performance    │────▶│ Performance     │────▶│ Permutation   │────▶│ Visualizations  │
│ Metrics        │     │ Distribution    │     │ Distribution  │     │ & P-values      │
└────────────────┘     └─────────────────┘     └───────────────┘     └─────────────────┘
```

## Creating Custom Strategies

To create a custom strategy:

1. Create a new strategy file in `src/strategies/`
2. Inherit from the base `Strategy` class
3. Implement the required methods (setup, process_data, generate_signals)
4. Register your strategy in the `registry.py` file

Example:

```python
from strategies.strategy import Strategy
from strategies.registry import register_strategy

class MyCustomStrategy(Strategy):
    def setup(self, parameters=None):
        # Initialize parameters
        self.param1 = parameters.get('param1', default_value)
        self.param2 = parameters.get('param2', default_value)
        
    def process_data(self, data):
        # Process market data
        # Calculate indicators or features needed for signal generation
        return processed_data
        
    def generate_signals(self, data):
        # Generate buy/sell signals based on processed data
        # Return a dictionary with signal information
        return signals

# Register the strategy with version information
register_strategy("MyCustomStrategy", MyCustomStrategy, "1.0.0")
```

## Parameter Optimization

The system supports parameter optimization using grid search to find the optimal parameter set for a given strategy:

1. Create a parameter grid file in `input/parameter_grids/` that defines the parameter space
2. Run the optimization workflow to test different parameter combinations
3. Analyze the results to identify the best-performing parameter set

Example parameter grid file (`my_strategy_grid.json`):

```json
{
  "param1": [10, 20, 30, 40, 50],
  "param2": [0.01, 0.02, 0.03, 0.04, 0.05],
  "param3": [true, false]
}
```

## Performance Evaluation

The system generates comprehensive evaluation metrics and visualizations:

- Equity curve with drawdown analysis
- Trade distribution statistics
- Monte Carlo confidence intervals
- P-value calculations for strategy robustness
- Comparison visualizations of original vs. permuted performance

### Key Metrics

- **Return Metrics**: Total return, annualized return, risk-adjusted return
- **Risk Metrics**: Max drawdown, volatility, Sharpe ratio, Sortino ratio
- **Trade Metrics**: Win rate, profit factor, average win/loss, max consecutive wins/losses
- **Statistical Metrics**: P-values, confidence intervals, probability of profit

## Advanced Features

### Enhanced Visualizations

The backtester includes advanced visualization capabilities that can be enabled with the `--enhanced-plots` flag:

- Monte Carlo simulation paths with confidence intervals
- Return distribution histograms with key statistics
- Drawdown analysis visualizations
- Comprehensive dashboard with combined metrics

### Walk-Forward Testing

The system supports walk-forward testing to validate strategy performance through time:

```bash
python src/workflows/cli.py --workflow walkforward \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --window-size 180 \
    --step-size 60 \
    --verbose
```

## Containerization and Frontend Integration

The system now has a fully integrated setup with the frontend application in the same repository:

1. **Single Repository**: Both backtester engine and frontend UI are in one repository
2. **Unified Docker Setup**: Single Docker configuration that runs both components
3. **Simplified Configuration**: Streamlined setup and configuration
4. **Improved Development Workflow**: Live editing of frontend files with volume mounting

The integrated frontend UI provides:
- Strategy selection and configuration
- Workflow management
- Parameter adjustments
- Results visualization
- Performance metrics display

For details on the Docker setup, see the [Docker README](docker/README.md).

## Key Considerations

1. **Data Quality**: Ensure your input data is clean and properly formatted
2. **Parameter Sensitivity**: Use Monte Carlo tests to evaluate parameter sensitivity
3. **Statistical Significance**: Focus on p-values to assess strategy robustness
4. **Look-Ahead Bias**: Avoid using future data in strategy logic
5. **Execution Costs**: Include realistic commission and slippage models in backtests
6. **Sample Size**: Ensure sufficient data for statistically significant results
7. **Out-of-Sample Testing**: Validate strategies on data not used during optimization

## Common Troubleshooting

### General Issues

- **Missing Data Error**: Ensure you've run `data_setup.py` first to create stock data files
- **Parameter File Not Found**: Check that your strategy's parameter file exists in `input/parameters/`
- **Visualization Issues**: Use the `--enhanced-plots` flag for improved visualizations
- **Multi-dimensional Indexing Error**: May occur with some pandas/numpy operations, ensure data is converted to numpy arrays before advanced indexing

### Docker-Specific Issues

- **Container Won't Start**: 
  - Check if port 5000 is already in use by another application
  - Verify Docker service is running with `docker info`
  - Ensure you have enough disk space with `docker system df`

- **Volume Mount Problems**: 
  - Ensure your directory structure matches the expected layout
  - Check file permissions on input/output directories (should be readable/writable)
  - On Windows, verify path format in docker-compose.yml is correct

- **Frontend Not Loading**: 
  - Check container logs with `docker-compose -f docker/docker-compose.yml logs`
  - Verify the container health with `docker ps` (should show "healthy" status)
  - Try accessing http://localhost:5000 in a different browser

- **Changes Not Reflecting**: 
  - Some changes require container rebuild - use `docker-compose -f docker/docker-compose.yml up --build`
  - Verify volume mounts are correctly set up in docker-compose.yml
  - Check that the BASE_DIR environment variable is correctly set

- **Permission Denied Errors**:
  - Run `chmod -R 777 input output logs cache frontend/temp frontend/output` to grant full permissions
  - On Linux/Mac, you may need to run Docker commands with sudo

- **Slow Performance**:
  - Increase Docker resource allocation (memory/CPU) in Docker Desktop settings
  - Consider moving volume mounts to fast storage (SSD instead of network drives)

## License

This project is licensed under the MIT License - see the LICENSE file for details.