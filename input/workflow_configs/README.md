# Workflow Configuration Files

This directory contains configuration files for running trading strategy workflows.

## Usage

Run a workflow using a config file:

```bash
python src/workflows/unified_workflow.py input/workflow_configs/simple_backtest_config.json
```

Or using the CLI:

```bash
python src/workflows/cli.py --config input/workflow_configs/simple_backtest_config.json
```

## Configuration File Structure

The configuration file uses JSON format with the following structure:

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
      "param_file": "input/parameters/ma_crossover_params.json",   // Optional: path to parameter file
      "grid_file": "input/parameter_grids/ma_crossover_grid.json", // Optional: path to grid file
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
    },
    "AnotherStrategy": {               // Additional strategies to run
      // Strategy-specific configuration
    }
  }
}
```

## Configuration Options

### Workflow Types

- `simple`: Basic backtest with a single parameter set
- `optimization`: Parameter grid search to find optimal parameters
- `monte_carlo`: Monte Carlo simulations based on historical data
- `walkforward`: Walk-forward testing with period-by-period optimization
- `complete`: Comprehensive analysis combining optimization, Monte Carlo, and walk-forward testing

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Start date for backtest | "2020-01-01" |
| `end_date` | End date for backtest | "2023-12-31" |
| `data_dir` | Directory for input data | "input" |
| `tickers` | List of ticker symbols | ["SPY"] |
| `initial_capital` | Initial capital for backtest | 100000.0 |
| `commission` | Commission rate | 0.001 |
| `plot` | Generate plots | false |
| `verbose` | Enable verbose logging | false |

### Workflow-Specific Parameters

#### Optimization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_trials` | Number of optimization trials | 50 |
| `optimization_metric` | Metric to optimize | "sharpe_ratio" |
| `max_combinations` | Maximum parameter combinations to test | null |
| `keep_all_results` | Save results of all parameter configurations | false |

#### Monte Carlo

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_simulations` | Number of Monte Carlo simulations | 100 |
| `enhanced_plots` | Generate enhanced visualization dashboard | false |
| `keep_permuted_data` | Keep permuted data after simulation | false |

#### Walk-forward

| Parameter | Description | Default |
|-----------|-------------|---------|
| `window_size` | Window size in trading days | 252 |
| `step_size` | Step size in trading days | 63 |
| `in_sample_ratio` | Ratio of in-sample data to the total window (0.0-1.0) | 0.7 |

## Example Files

- `simple_workflow_config.json`: Basic backtest with a single parameter set
- `optimization_workflow_config.json`: Parameter optimization with grid search
- `monte_carlo_workflow_config.json`: Monte Carlo simulations for risk analysis
- `walkforward_workflow_config.json`: Walk-forward testing for robustness testing
- `complete_config.json`: Comprehensive workflow that combines multiple analysis types
- `auction_market_config.json`: Example configuration for the auction market strategy

## Standard Output Format

### Equity Curve CSV Format

All workflows produce a standardized `equity_curve.csv` file with the following columns:

| Column | Description |
|--------|-------------|
| `date` | Trading date in YYYY-MM-DD format |
| `equity` | Total equity value of the portfolio |
| `pnl` | Daily profit and loss |
| `cumulative_pnl` | Cumulative profit and loss from start date |
| `returns` | Daily percentage returns |
| `cumulative_returns` | Cumulative percentage returns from start date |
| `drawdown` | Current drawdown as a percentage from peak |
| `positions` | Number of open positions (if available) |
| `cash` | Cash balance (if available) |
| `market_value` | Market value of holdings (if available) |

This standardized format is consistent across all workflow types, making it easier to compare results from different backtests.