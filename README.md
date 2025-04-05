# Trading Strategy Backtester Framework

This project provides a Python-based framework for backtesting, optimizing, and evaluating trading strategies using historical market data. It specializes in Monte Carlo simulations for robust strategy validation.

## Features

- **Monte Carlo Testing:** Core feature for robust strategy validation using the `DirectMonteCarloTest` class
  - Data permutation techniques for statistical validation
  - P-value calculations for key performance metrics
  - Parameter optimization and trade generation enhancement
  
- **Backtesting Engine:** Utilizes the `backtrader` library for event-driven backtesting
  - Customizable strategy implementation
  - Detailed performance metrics and trade logging
  - Portfolio value tracking and equity curve generation

- **Strategy Support:** 
  - Built-in strategies (`SimpleStock`, `MACrossover`, `AuctionMarket`, `MultiPosition`)
  - Strategy registry for custom implementations
  - Parameter variation during testing

- **Performance Metrics:** 
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Trade Count validation

- **Parallel Processing:** 
  - Multi-core support for faster Monte Carlo simulations
  - Configurable worker count
  - Parallel execution of backtests with multiple parameter combinations
  - Automatic CPU core detection and allocation

## Monte Carlo Simulation

The Monte Carlo simulation framework creates synthetic but statistically similar market conditions to test how a trading strategy performs under different scenarios. This helps determine if a strategy's performance is due to actual edge or random chance.

### Overall Process Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌──────────────┐
│ Load Stock  │────▶│ Run Original │────▶│ Generate Data │────▶│ Run Monte   │────▶│ Analyze      │
│ Data        │     │ Backtest     │     │ Permutations  │     │ Carlo Tests │     │ Results      │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘     └──────────────┘
```

### Data Loading & Preparation

```
┌─────────────┐     ┌───────────────┐     ┌───────────────┐
│ Stock Data  │────▶│ Data Cleaning │────▶│ Format        │
│ CSV Files   │     │ & Validation  │     │ for Backtrader│
└─────────────┘     └───────────────┘     └───────────────┘
```

### Monte Carlo Simulation Process

```
┌────────────┐                                     ┌─────────────┐
│ Original   │                                     │ Statistical │
│ Backtest   │──┐                                  │ Analysis    │
└────────────┘  │                                  └─────────────┘
                │                                         ▲
                ▼                                         │
┌─────────────────────────────────────────────────────────────────────┐
│                   Monte Carlo Permutations                           │
│                                                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐ │
│  │Permutation │    │Permutation │    │Permutation │    │   ...    │ │
│  │    #1      │    │    #2      │    │    #3      │    │          │ │
│  └────────────┘    └────────────┘    └────────────┘    └──────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Permutation Methods

The code implements three different permutation methods to create synthetic market data:

```
1. Returns Permutation
   ┌───────────┐     ┌────────────┐     ┌───────────────┐
   │ Calculate │────▶│ Shuffle    │────▶│ Reconstruct   │
   │ Returns   │     │ Returns    │     │ Price Series  │
   └───────────┘     └────────────┘     └───────────────┘

2. Block Permutation
   ┌───────────┐     ┌────────────┐     ┌───────────────┐
   │ Create    │────▶│ Shuffle    │────▶│ Reconstruct   │
   │ Blocks    │     │ Blocks     │     │ Price Series  │
   └───────────┘     └────────────┘     └───────────────┘

3. Stationary Bootstrap
   ┌────────────┐     ┌────────────────────┐     ┌───────────────┐
   │ Generate   │────▶│ Sample Blocks with │────▶│ Reconstruct   │
   │ Block Size │     │ Random Lengths     │     │ Price Series  │
   └────────────┘     └────────────────────┘     └───────────────┘
```

### Parameter Optimization & Trade Generation

The framework ensures that every Monte Carlo permutation generates meaningful trades through:

```
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Initial        │────▶│ Adaptive        │────▶│ Aggressive      │
│ Parameters     │     │ Parameters Per  │     │ Parameters      │
│                │     │ Permutation     │     │ (If No Trades)  │
└────────────────┘     └─────────────────┘     └─────────────────┘
```

### Statistical Analysis

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Original       │     │ Calculate       │     │ Compare with  │     │ Generate        │
│ Performance    │────▶│ Performance     │────▶│ Permutation   │────▶│ Visualizations  │
│ Metrics        │     │ Distribution    │     │ Distribution  │     │ & P-values      │
└────────────────┘     └─────────────────┘     └───────────────┘     └─────────────────┘
```

### Visualization Suite

The framework generates comprehensive visualizations:

```
┌─────────────────┐     ┌────────────────┐     ┌────────────────┐
│ Distribution    │     │ Equity Curve   │     │ Price Path     │
│ Plots Per Metric│─────│ Visualizations │─────│ Comparisons    │
│                 │     │                │     │                │
└─────────────────┘     └────────────────┘     └────────────────┘
```

### Key Concepts

1. **Permutation Testing**: By randomly shuffling historical data while preserving its statistical properties, the system tests if strategy performance is due to actual edge or random chance.

2. **Multiple Permutation Methods**: Different shuffling approaches preserve different market characteristics.

3. **Parameter Optimization**: Automatically finds and reports the best parameters across permutations based on composite performance scores.

4. **Adaptive Parameters**: Ensures trades are generated for every permutation by adjusting parameters and implementing fallback mechanisms.

5. **P-value Calculation**: The proportion of permutations that outperform the original strategy indicates statistical significance.

6. **Trade Validation**: Ensures that strategies actually generate trades and tracks trade counts as a key metric.

## Directory Structure

```
.
├── src/
│   ├── monte_carlo/       # Direct Monte Carlo implementation
│   │   └── direct_monte_carlo.py  # Core Monte Carlo testing logic
│   ├── strategies/        # Trading strategy implementations
│   │   ├── simple_stock.py
│   │   └── ma_crossover.py
│   └── utils/             # Utility functions
├── input/                 # Stock data input files
├── output/                # Test results and visualizations
│   └── monte_carlo_test_*/  # Timestamped Monte Carlo test results
├── run_complete_monte_carlo.py  # Main entry point script
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT license
└── README.md              # This documentation
```

## Installation

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
   pip install backtrader # Add backtrader explicitly
   ```
   *Note: Ensure `backtrader`, `pandas`, `numpy`, and `matplotlib` are installed.*

## Usage

The primary entry point is `run_complete_monte_carlo.py` for running Monte Carlo tests:

```bash
python run_complete_monte_carlo.py \
    --strategy SimpleStock \
    --tickers AAPL \
    --num-permutations 100 \
    --num-cores 4 \
    --verbose
```

**Common Arguments:**

* `--strategy`: Name of the strategy (e.g., `SimpleStock`, `MACrossover`).
* `--tickers`: Comma-separated list of stock tickers (e.g., `AAPL,MSFT`).
* `--start-date`: Start date for data (YYYY-MM-DD). Default uses all available data.
* `--end-date`: End date for data (YYYY-MM-DD). Default uses all available data.
* `--in-sample-ratio`: Ratio of data to use for in-sample period (0.0 to 1.0). Default is 0.7.
* `--num-permutations`: Number of Monte Carlo permutations to run. Default is 100.
* `--num-cores`: Number of CPU cores for parallel processing. Default uses all available cores except one.
* `--output-dir`: Specify a custom output directory. Default is timestamped directory.
* `--verbose`: Enable detailed logging.

## Parallel Processing

The framework supports parallel processing for improved performance:

1. **Monte Carlo Simulations**: By default, Monte Carlo simulations run on multiple CPU cores:
   ```bash
   python run_complete_monte_carlo.py --strategy SimpleStock --num-simulations 1000 --num-cores 8
   ```

2. **Batch Parameter Testing**: Test multiple parameter combinations in parallel:
   ```bash
   python run_parameter_optimization.py --strategy MACrossover --num-cores 8
   ```

3. **Automatic Core Detection**: If `--num-cores` is not specified, the system automatically uses all available cores except one to prevent system slowdown.

4. **Core Management**: CPU core allocation can be controlled via:
   * `--num-cores` command-line argument
   * `num_workers` parameter in direct API usage
   * Environment variable `TRADING_BACKTEST_CORES`

## Output Structure

Results are saved in timestamped directories like `output/monte_carlo_test_YYYYMMDD_HHMMSS/`. Common outputs include:

* **Original Backtest Results:**
  * `original_results.json`: Performance metrics for original backtest
  * `trade_log_original.csv`: Log of all trades from original backtest
  * `equity_curve.csv`: Portfolio value over time for original backtest

* **Permutation Results:**
  * Individual permutation results in separate directories
  * Parameter variations used for each permutation
  * Best parameters found across all permutations

* **Visualizations:**
  * Distribution plots for key metrics (PNG files)
  * Equity curves comparing original vs. permutations
  * Price path visualizations
  * Statistical significance indicators

* **Summary Reports:**
  * P-values for performance metrics
  * Best parameters recommendation
  * Overall strategy robustness assessment

## Contributing

[Optional: Add guidelines for contribution]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 