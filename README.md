# Trading Strategy Backtester Framework

This project provides a Python-based framework for backtesting, optimizing, and evaluating trading strategies using historical market data. It integrates several key testing methodologies, including simple backtesting, parameter optimization, walk-forward analysis, and Monte Carlo simulations.

## Features

-   **Unified Workflow:** Provides a single command-line interface (`unified_workflow.py`) to run different testing scenarios.
-   **Backtesting Engine:** Utilizes the `backtrader` library for robust event-driven backtesting.
-   **Strategy Support:** Includes several example strategies (e.g., `SimpleStock`, `MACrossover`, `AuctionMarket`, `MultiPosition`) and a registry for adding custom strategies.
-   **Parameter Optimization:** Finds optimal strategy parameters using in-sample data.
-   **Walk-Forward Testing:** Evaluates strategy robustness by testing optimized parameters on subsequent out-of-sample periods.
-   **Monte Carlo Simulation:** Assesses strategy robustness against variations in market data using data permutation techniques (via `DirectMonteCarloTest`). Calculates p-values for key metrics.
-   **Performance Metrics:** Calculates standard performance metrics (Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor).
-   **Detailed Logging:** Generates comprehensive logs, including trade logs and portfolio value tracking.
-   **Parallel Processing:** Leverages multiple CPU cores for optimization and Monte Carlo simulations.

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

### Statistical Analysis

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Original       │     │ Calculate       │     │ Compare with  │     │ Generate        │
│ Performance    │────▶│ Performance     │────▶│ Permutation   │────▶│ Visualizations  │
│ Metrics        │     │ Distribution    │     │ Distribution  │     │ & P-values      │
└────────────────┘     └─────────────────┘     └───────────────┘     └─────────────────┘
```

### Parameter Optimization

```
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐
│ Vary Strategy  │────▶│ Evaluate        │────▶│ Calculate       │────▶│ Identify Best │
│ Parameters     │     │ Performance     │     │ Composite       │     │ Parameter Set │
│ Per Permutation│     │ Per Parameter   │     │ Score           │     │               │
└────────────────┘     └─────────────────┘     └─────────────────┘     └───────────────┘
```

### Key Concepts

1. **Permutation Testing**: By randomly shuffling historical data while preserving its statistical properties, the system tests if strategy performance is due to actual edge or random chance.

2. **Multiple Permutation Methods**: Different shuffling approaches (returns, blocks, stationary bootstrap) preserve different market characteristics.

3. **Parameter Robustness**: Testing parameters across many market conditions helps find settings that work in various environments.

4. **P-value Calculation**: The proportion of permutations that outperform the original strategy indicates statistical significance.

5. **In-Sample vs Out-of-Sample**: The framework can test on different time periods to assess strategy robustness.

## Directory Structure (Simplified)

```
.
├── src/
│   ├── engine/          # Core backtesting, optimization, and testing logic
│   ├── strategies/      # Trading strategy implementations
│   ├── workflows/       # High-level workflow scripts (e.g., unified_workflow.py)
│   ├── monte_carlo/     # Direct Monte Carlo simulation implementation
│   └── runners/         # Helper scripts for specific runs
├── input/               # Input data (e.g., stock_data.csv - requires user setup)
├── output/              # Directory for backtest results, logs, and plots
├── docs/                # (Potentially outdated) Detailed documentation
├── requirements.txt     # Python package dependencies
└── README.md            # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd trading-strategy-backtester
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
    pip install backtrader # Add backtrader explicitly
    ```
    *Note: The `requirements.txt` might be incomplete. Ensure `backtrader`, `pandas`, `numpy`, and `matplotlib` are installed.*

## Usage

The primary entry point is `src/workflows/unified_workflow.py`. You can run different workflows using the `--workflow-type` argument.

**Common Arguments:**

*   `--strategy`: Name of the strategy (e.g., `SimpleStock`, `MACrossover`).
*   `--tickers`: Comma-separated list of stock tickers (e.g., `AAPL,MSFT`).
*   `--start-date`: Start date (YYYY-MM-DD).
*   `--end-date`: End date (YYYY-MM-DD).
*   `--num-cores`: Number of CPU cores for parallel tasks (optional).
*   `--output-dir`: Specify a custom output directory (optional).
*   `--verbose`: Enable detailed logging (optional).

**Example Workflows:**

1.  **Run a Simple Backtest:**
    Uses default parameters for the specified strategy over the entire date range.
    ```bash
    python -m src.workflows.unified_workflow --workflow-type simple --strategy SimpleStock --tickers AAPL --start-date 2020-01-01 --end-date 2022-12-31
    ```

2.  **Run Parameter Optimization:**
    Optimizes strategy parameters on an in-sample period (first 70% of data by default).
    ```bash
    python -m src.workflows.unified_workflow --workflow-type optimize --strategy MACrossover --tickers MSFT --start-date 2019-01-01 --end-date 2022-12-31 --num-cores 4
    ```
    *(Requires a parameter file, e.g., `src/strategies/params/MACrossover_params.json`, to define parameter ranges. See code/docs for details).*

3.  **Run Walk-Forward Analysis:**
    Performs optimization on rolling in-sample windows and tests on subsequent out-of-sample windows.
    ```bash
    python -m src.workflows.unified_workflow --workflow-type walkforward --strategy SimpleStock --tickers AAPL,GOOG --start-date 2018-01-01 --end-date 2022-12-31 --num-cores 4
    ```
    *(Requires parameter file. Configure walk-forward parameters like window sizes within the script or via args if implemented).*

4.  **Run Monte Carlo Simulation:**
    Uses the best parameters (can be found via optimization or defaults) and runs Monte Carlo permutations on out-of-sample data.
    ```bash
    # Assumes best parameters are found or defaults are used
    python -m src.workflows.unified_workflow --workflow-type montecarlo --strategy MACrossover --tickers AAPL --start-date 2019-01-01 --end-date 2022-12-31 --num-permutations 100 --num-cores 4
    ```

5.  **Run a Complete Workflow (Optimization + Walk-Forward + Monte Carlo):**
    Executes optimization, then walk-forward, and finally Monte Carlo using the best parameters found.
    ```bash
    python -m src.workflows.unified_workflow --workflow-type complete --strategy SimpleStock --tickers AAPL --start-date 2019-01-01 --end-date 2022-12-31 --num-permutations 100 --num-cores 4
    ```
    *(Requires parameter file).*

## Output

Results, logs, and plots are saved in timestamped subdirectories within the `output/` folder by default, or in the specified `--output-dir`. Common outputs include:
-   Performance metrics (JSON or text files)
-   Portfolio value charts (PNG)
-   Trade logs (CSV)
-   Monte Carlo result distributions (plots and data)

## Contributing

[Optional: Add guidelines for contribution]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 