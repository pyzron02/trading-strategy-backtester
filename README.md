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

[Specify your license - e.g., MIT, GPL-3.0]
*(The previous README mentioned GPL-3.0)* 