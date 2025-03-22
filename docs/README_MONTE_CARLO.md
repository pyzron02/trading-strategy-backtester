# Direct Monte Carlo Testing Framework

This framework provides a robust implementation for testing trading strategies using Monte Carlo simulations. It allows for statistical evaluation of strategy performance by comparing the original backtest results with permutations of the price data.

## Features

- **Direct Backtrader Integration**: Uses Backtrader's native functionality for reliable backtesting
- **Data Permutation**: Shuffles returns or price blocks in out-of-sample data while preserving in-sample data
- **Statistical Analysis**: Calculates p-values and generates distribution plots for key metrics
- **Equity Curve Tracking**: Records and saves the equity curve for each test
- **Multiple Permutation Types**: Supports both returns-based and block-based permutations
- **Robust Error Handling**: Safely processes analyzer results from Backtrader

## Included Strategies

1. **SimpleStock Strategy**:
   - Buys when price is above SMA
   - Sells when price is below SMA
   - Includes proper warmup periods for stable indicators

2. **MACrossover Strategy**:
   - Uses two moving averages (fast and slow)
   - Generates buy signals when fast MA crosses above slow MA
   - Generates sell signals when fast MA crosses below slow MA
   - Includes proper warmup periods for stable indicators

## Usage

```bash
python direct_monte_carlo.py --strategy MACrossover --tickers AAPL --num_permutations 5 --in_sample_start 2015-01-01 --in_sample_end 2019-12-31
```

### Parameters:

- `--strategy`: Name of the strategy (SimpleStock or MACrossover)
- `--tickers`: Comma-separated list of tickers
- `--num_permutations`: Number of permutations to run
- `--in_sample_start`: In-sample period start date (YYYY-MM-DD)
- `--in_sample_end`: In-sample period end date (YYYY-MM-DD)
- `--out_sample_start`: Out-of-sample period start date (YYYY-MM-DD)
- `--out_sample_end`: Out-of-sample period end date (YYYY-MM-DD)

## Monte Carlo Analysis

The framework calculates p-values for key metrics by comparing the original backtest results with the distribution of permutation results:

- **Sharpe Ratio**: Measures risk-adjusted return
- **Total Return**: Overall return of the strategy
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Proportion of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses

A p-value of less than 0.05 indicates that the strategy's performance is statistically significant compared to random permutations, suggesting that the strategy has genuine predictive power rather than benefiting from lucky market conditions.

## Output

The framework generates the following outputs:

1. Test parameters JSON file
2. Original backtest results
3. Permutation test results
4. Statistical analysis of all tests
5. Distribution plots for each metric
6. Equity curves for the original and permutation tests
7. Trade logs for the original strategy and each permutation

All outputs are saved in the designated output directory with a timestamp.

## Trade Logging

The Monte Carlo framework now includes detailed trade logging for both the original strategy and each permutation. This allows for deeper analysis of how trading behavior changes across different market conditions.

### Trade Log Features:

- **Comprehensive Statistics**: Records trade counts, profit/loss metrics, and trade durations
- **Trade Classification**: Breaks down trades by type (win/loss) and direction (long/short)
- **Per-Permutation Analysis**: Separate trade logs for each Monte Carlo permutation
- **Summary File**: A JSON summary file that lists paths to all trade logs

### Trade Log Format:

Each trade log is a CSV file with the following columns:
```
Type, Direction, Count, PnL Total, PnL Avg, PnL Max, Length Avg, Length Max
```

Where:
- **Type**: Category of trade (All, Won, Lost)
- **Direction**: Trade direction (All, Long, Short)
- **Count**: Number of trades in this category
- **PnL Total**: Total profit/loss for these trades
- **PnL Avg**: Average profit/loss per trade
- **PnL Max**: Maximum profit/loss from a single trade
- **Length Avg**: Average number of bars a trade was held
- **Length Max**: Maximum number of bars a trade was held

### Accessing Trade Logs:

When running the Monte Carlo workflow, trade logs are stored in:
- `{output_dir}/original/trade_log_original.csv` for the original strategy
- `{output_dir}/permutation_{i}/trade_log_permutation_{i}.csv` for each permutation

A summary file `trade_log_summary.json` is also created in the output directory, providing an index to all trade logs.

## Implementation Benefits

This direct Monte Carlo implementation offers several advantages:

1. **Stability**: Avoids index errors and other issues that can occur in more complex frameworks
2. **Flexibility**: Easily accommodates different strategies and permutation methods
3. **Transparency**: Clear processing of results with detailed output
4. **Visual Analysis**: Generates histograms to visually assess strategy performance
5. **Trade Analysis**: Detailed trade logs for deeper understanding of strategy behavior across permutations

## Future Enhancements

- Add support for more permutation techniques
- Implement walk-forward optimization
- Add more trading strategies
- Enhance visualization capabilities
- Expand trade logging with more detailed metrics and transaction-level data 