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

```
.
├── src/
│   ├── engine/             # Core backtesting engine components
│   │   ├── run_backtest.py       # Main backtesting engine
│   │   ├── evaluate_performance.py # Performance metric calculations
│   │   ├── data_management.py    # Data loading and preprocessing
│   │   ├── parameter_management.py # Strategy parameter handling
│   │   ├── results_management.py # Output and results handling
│   │   ├── parallel_testing.py   # Parallel execution support
│   │   ├── smart_cache.py        # Caching system for performance
│   │   └── logging_system.py     # Logging infrastructure
│   │
│   ├── data_preprocessing/ # Data preparation tools
│   │   ├── data_setup.py         # Data acquisition and initial setup
│   │   └── feature_engineering.py # Feature creation for strategies
│   │
│   ├── evaluators/         # Performance evaluation utilities
│   │   ├── check_strategies.py   # Strategy validation
│   │   ├── test_strategies.py    # Strategy testing
│   │   └── generate_monte_carlo_plots.py # MC visualization
│   │
│   ├── monte_carlo/        # Monte Carlo simulation implementation
│   │   ├── monte_carlo_analysis.py # Core MC analysis engine
│   │   ├── trade_based_monte_carlo.py # Trade-based MC implementation
│   │   ├── visualizations.py    # Enhanced visualization components
│   │   ├── strategies.py        # MC-specific strategy handling
│   │   └── utilities.py         # MC helper functions
│   │
│   ├── optimizers/         # Parameter optimization components
│   │   ├── run_parameter_optimization.py # Parameter optimization
│   │   └── create_optimization_summary.py # Results documentation
│   │
│   ├── strategies/         # Trading strategy implementations
│   │   ├── strategy.py           # Base Strategy class
│   │   ├── ma_crossover.py       # Moving Average Crossover strategy
│   │   ├── auction_market_strategy.py # Auction Market strategy
│   │   ├── multi_position_strategy.py # Multi-position strategy
│   │   ├── simple_stock_strategy.py # Simple stock trading strategy
│   │   └── registry.py          # Strategy registration system
│   │
│   └── workflows/          # High-level workflow orchestrators
│       ├── simple_workflow.py    # Single backtest workflow
│       ├── optimization_workflow.py # Parameter optimization workflow
│       ├── monte_carlo_workflow.py # Monte Carlo simulation workflow
│       ├── complete_workflow.py  # Combined workflow (opt + backtest + MC)
│       ├── unified_workflow.py   # Entry point for all workflows
│       └── cli.py               # Command line interface
│
├── input/                  # Stock data input files
│   ├── stock_data.csv          # Historical price data
│   ├── parameters/             # Strategy parameter files
│   ├── parameter_grids/        # Parameter optimization grid definitions
│   └── workflow_configs/       # Workflow configuration files
│
├── output/                 # Test results and visualizations
├── tests/                  # Automated tests
├── run_trade_monte_carlo.py # Monte Carlo workflow entry point
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

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-strategy-backtester
   ```

2. **Build and run using Docker:**
   ```bash
   docker build -t trading-backtester .
   docker run -v $(pwd)/input:/app/trading-strategy-backtester/input \
              -v $(pwd)/output:/app/trading-strategy-backtester/output \
              -v $(pwd)/logs:/app/trading-strategy-backtester/logs \
              trading-backtester
   ```

3. **Or use Docker Compose:**
   ```bash
   docker-compose up
   ```

### Method 3: Integration with Frontend

For integration with a frontend application, see the [Containerization Guide](README_CONTAINER.md).

1. **Create project directory structure:**
   ```bash
   mkdir my-trading-app
   cd my-trading-app
   git clone <backtester-repo-url> trading-strategy-backtester
   git clone <frontend-repo-url> frontend
   ```

2. **Set up Docker Compose:**
   Copy the parent Docker Compose file to your project root:
   ```bash
   cp trading-strategy-backtester/docker-compose.parent.yml docker-compose.yml
   ```

3. **Run both services:**
   ```bash
   docker-compose up
   ```

## Usage

### Configuration Files

The most flexible way to use the backtester is with configuration files:

```bash
python src/workflows/cli.py --config input/workflow_configs/simple_backtest_config.json
```

Or directly with the unified workflow runner:

```bash
python src/workflows/unified_workflow.py input/workflow_configs/multi_strategy_config.json
```

Example configuration files are available in the `input/workflow_configs/` directory.

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
docker-compose exec backtester python src/workflows/cli.py --workflow simple \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01
```

Or configure the command in docker-compose.yml:

```yaml
services:
  backtester:
    # ... other settings ...
    command: python src/workflows/cli.py --workflow simple --strategy MACrossover --tickers AAPL
```

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

The system now supports containerization for easy deployment and integration with frontend applications:

1. **Centralized Path Management**: All paths are now relative, making the application containerization-friendly
2. **Docker Configuration**: Ready-to-use Dockerfile and docker-compose.yml files
3. **Frontend Integration**: Configured for easy integration with a frontend application in a containerized environment

For details on containerization and integration with a frontend application, see the [Containerization Guide](README_CONTAINER.md).

## Key Considerations

1. **Data Quality**: Ensure your input data is clean and properly formatted
2. **Parameter Sensitivity**: Use Monte Carlo tests to evaluate parameter sensitivity
3. **Statistical Significance**: Focus on p-values to assess strategy robustness
4. **Look-Ahead Bias**: Avoid using future data in strategy logic
5. **Execution Costs**: Include realistic commission and slippage models in backtests
6. **Sample Size**: Ensure sufficient data for statistically significant results
7. **Out-of-Sample Testing**: Validate strategies on data not used during optimization

## Common Troubleshooting

- **Missing Data Error**: Ensure you've run `data_setup.py` first to create stock data files
- **Parameter File Not Found**: Check that your strategy's parameter file exists in `input/parameters/`
- **Visualization Issues**: Use the `--enhanced-plots` flag for improved visualizations
- **Multi-dimensional Indexing Error**: May occur with some pandas/numpy operations, ensure data is converted to numpy arrays before advanced indexing
- **Docker Path Issues**: If using Docker and experiencing path problems, ensure volume mounts are configured correctly and the BASE_DIR environment variable is set

## License

This project is licensed under the MIT License - see the LICENSE file for details.