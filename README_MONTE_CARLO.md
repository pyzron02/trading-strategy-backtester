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

All outputs are saved in the designated output directory with a timestamp.

## Implementation Benefits

This direct Monte Carlo implementation offers several advantages:

1. **Stability**: Avoids index errors and other issues that can occur in more complex frameworks
2. **Flexibility**: Easily accommodates different strategies and permutation methods
3. **Transparency**: Clear processing of results with detailed output
4. **Visual Analysis**: Generates histograms to visually assess strategy performance

## Future Enhancements

- Add support for more permutation techniques
- Implement walk-forward optimization
- Add more trading strategies
- Enhance visualization capabilities 