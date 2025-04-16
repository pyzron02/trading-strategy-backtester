# Trading Strategy Backtester

A robust framework for backtesting, optimizing, and evaluating trading strategies using historical market data, with a focus on Monte Carlo simulations for strategy validation.

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
│   ├── data_preprocessing/ # Data preparation tools
│   ├── evaluators/         # Performance evaluation utilities
│   ├── monte_carlo/        # Monte Carlo simulation implementation
│   ├── optimizers/         # Parameter optimization components
│   ├── runners/            # Single-purpose script runners
│   ├── strategies/         # Trading strategy implementations
│   ├── utils/              # Utility functions
│   └── workflows/          # High-level workflow orchestrators
├── input/                  # Stock data input files
├── output/                 # Test results and visualizations
├── tests/                  # Automated tests
├── run_trade_monte_carlo.py # Monte Carlo workflow entry point
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT license
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
   ```

## Usage

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
    --verbose
```

### Optimization Workflow

To optimize strategy parameters:

```bash
python src/workflows/optimization_workflow.py \
    --strategy MACrossover \
    --tickers AAPL \
    --start-date 2020-01-01 \
    --end-date 2025-01-01 \
    --n-trials 50 \
    --verbose
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

class MyCustomStrategy(Strategy):
    def setup(self, parameters=None):
        # Initialize parameters
        self.param1 = parameters.get('param1', default_value)
        
    def process_data(self, data):
        # Process market data
        
    def generate_signals(self, data):
        # Generate buy/sell signals
        return signals
```

## Performance Evaluation

The system generates comprehensive evaluation metrics and visualizations:

- Equity curve with drawdown analysis
- Trade distribution statistics
- Monte Carlo confidence intervals
- P-value calculations for strategy robustness
- Comparison visualizations of original vs. permuted performance

## Key Considerations

1. **Data Quality**: Ensure your input data is clean and properly formatted
2. **Parameter Sensitivity**: Use Monte Carlo tests to evaluate parameter sensitivity
3. **Statistical Significance**: Focus on p-values to assess strategy robustness
4. **Look-Ahead Bias**: Avoid using future data in strategy logic
5. **Execution Costs**: Include realistic commission and slippage models in backtests

## License

This project is licensed under the MIT License - see the LICENSE file for details. 