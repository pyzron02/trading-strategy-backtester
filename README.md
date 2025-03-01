# Trading Strategy Backtesting Framework

## Overview

This codebase provides a comprehensive framework for backtesting trading strategies, evaluating their performance, and comparing multiple strategies. It's designed to help traders and quantitative analysts assess the effectiveness of different trading approaches using historical market data.

## Features

- **Backtest Trading Strategies**: Run simulations of trading strategies against historical market data
- **Performance Evaluation**: Calculate key performance metrics such as returns, volatility, Sharpe ratio, and drawdowns
- **Visualization**: Generate visual representations of equity curves, drawdowns, and return distributions
- **Strategy Comparison**: Compare multiple strategies side-by-side to identify strengths and weaknesses
- **Benchmark Comparison**: Compare strategy performance against market benchmarks like the S&P 500

## Directory Structure

```
.
├── grok_code/
│   └── src/
│       ├── evaluate_performance.py  # Script for evaluating backtest results
│       └── compare_strategies.py    # Script for comparing multiple strategies
├── input/
│   └── stock_data.csv               # Historical market data
└── output/
    ├── YYYY-MM-DD_HH-MM-SS_StrategyName_portfolio/  # Strategy output directories
    │   ├── backtest_results.pkl     # Serialized backtest results
    │   ├── equity_curve.csv         # Portfolio value over time
    │   ├── trade_log.csv            # Record of all trades
    │   ├── results.txt              # Summary performance metrics
    │   ├── equity_curve.png         # Equity curve visualization
    │   ├── drawdown.png             # Drawdown visualization
    │   └── returns_histogram.png    # Returns distribution
    └── comparison/                  # Strategy comparison results
        ├── combined_equity_curves.png       # Combined equity curves
        ├── total_return_comparison.png      # Total return comparison
        ├── annual_return_comparison.png     # Annualized return comparison
        ├── sharpe_ratio_comparison.png      # Sharpe ratio comparison
        ├── max_drawdown_comparison.png      # Maximum drawdown comparison
        └── strategy_metrics_comparison.csv  # Tabular metrics comparison
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd trading-strategy-backtester
   ```

2. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn
   ```

## Usage Guide

### 1. Running a Backtest

This framework assumes you have a separate backtesting engine that generates the initial backtest results. The output of your backtest should be saved as a pickle file (`backtest_results.pkl`) in a directory with the naming convention:

```
output/YYYY-MM-DD_HH-MM-SS_StrategyName_portfolio/
```

The backtest should also generate:
- `equity_curve.csv`: A CSV file containing the portfolio value over time
- `trade_log.csv`: A CSV file containing the record of all trades

### 2. Evaluating Performance

After running a backtest, evaluate its performance using:

```
python grok_code/src/evaluate_performance.py output/YYYY-MM-DD_HH-MM-SS_StrategyName_portfolio/backtest_results.pkl
```

This will:
- Calculate key performance metrics (total return, annualized return, volatility, Sharpe ratio, maximum drawdown)
- Compare performance against the S&P 500 benchmark
- Generate visualizations (equity curve, drawdown, returns histogram)
- Save a summary of results to `results.txt`

### 3. Comparing Multiple Strategies

To compare the performance of multiple strategies:

```
python grok_code/src/compare_strategies.py output/comparison output/YYYY-MM-DD_HH-MM-SS_Strategy1_portfolio output/YYYY-MM-DD_HH-MM-SS_Strategy2_portfolio [...]
```

This will:
- Extract performance metrics from each strategy
- Generate combined equity curve visualization
- Create bar charts comparing key metrics
- Save a CSV file with all metrics for further analysis

## Example Workflow

1. Run backtests for multiple trading strategies
2. Evaluate each strategy's performance:
   ```
   python grok_code/src/evaluate_performance.py output/2025-02-28_18-09-08_MultiPosition_portfolio/backtest_results.pkl
   python grok_code/src/evaluate_performance.py output/2025-02-28_18-59-14_SimpleStock_portfolio/backtest_results.pkl
   ```
3. Compare the strategies:
   ```
   python grok_code/src/compare_strategies.py output/comparison output/2025-02-28_18-09-08_MultiPosition_portfolio output/2025-02-28_18-59-14_SimpleStock_portfolio
   ```
4. Review the comparison results in the `output/comparison/` directory

## Performance Metrics

The framework calculates the following key performance metrics:

- **Total Return**: The overall percentage return of the strategy
- **Annualized Return**: The return normalized to an annual basis
- **Annualized Volatility**: The standard deviation of returns on an annual basis
- **Sharpe Ratio**: The risk-adjusted return (excess return divided by volatility)
- **Maximum Drawdown**: The largest peak-to-trough decline in portfolio value

## Troubleshooting

### Common Issues

1. **Missing Files**: Ensure that each strategy directory contains the required files (`backtest_results.pkl`, `equity_curve.csv`)
2. **Directory Names**: Make sure to use the correct directory paths when running the scripts
3. **Data Format**: The equity curve CSV should have 'Date' and 'Value' columns
4. **Benchmark Data**: For benchmark comparison, ensure `input/stock_data.csv` exists with an 'SP500_Close' column

### Error Messages

- "Backtest results file not found": Check the path to your backtest results file
- "No results.txt file found": Run the evaluation script first before comparing strategies
- "No equity curve found": Ensure the equity_curve.csv file exists in the strategy directory

## Extending the Framework

### Adding New Metrics

To add new performance metrics:
1. Modify the `evaluate_performance.py` script to calculate the new metric
2. Update the `results.txt` output to include the new metric
3. Modify the `extract_metric` function in `compare_strategies.py` to extract the new metric
4. Add visualization for the new metric in `plot_metrics_comparison`

### Customizing Visualizations

The visualization settings can be customized by modifying the plotting functions in both scripts:
- `plot_equity_curve`, `plot_drawdown`, and `plot_returns_histogram` in `evaluate_performance.py`
- `plot_combined_equity_curves` and `plot_metrics_comparison` in `compare_strategies.py`

## License

[Specify your license information here]

## Contact

[Your contact information or how to report issues] 