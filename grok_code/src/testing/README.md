# Trading Strategy Testing Framework

This directory contains a comprehensive set of testing tools for validating trading strategies. These tests help ensure that your strategies are robust, statistically significant, and not just fitting to historical data.

## Available Tests

### 1. In-Sample Excellence

Optimizes strategy parameters on historical data to achieve strong performance metrics like high Sharpe ratios and profit factors.

```bash
python in_sample_excellence.py --strategy SimpleStock --tickers AAPL MSFT GOOGL --start_date 2015-01-01 --end_date 2019-12-31
```

### 2. In-Sample Monte Carlo Permutation Test

Compares strategy performance on real data to permuted data to check for data mining bias, ensuring results are not due to random noise.

```bash
python in_sample_monte_carlo.py --strategy SimpleStock --tickers AAPL MSFT GOOGL --start_date 2015-01-01 --end_date 2019-12-31 --num_permutations 100
```

### 3. Walk-Forward Test

Evaluates strategy performance on out-of-sample data to assess real-world applicability.

```bash
python walk_forward_test.py --strategy SimpleStock --tickers AAPL MSFT GOOGL --in_sample_start 2015-01-01 --in_sample_end 2019-12-31 --out_sample_start 2020-01-01 --out_sample_end 2021-12-31
```

### 4. Walk-Forward Monte Carlo Permutation Test

Validates out-of-sample performance by comparing it to permuted data, ensuring statistical significance.

```bash
python walk_forward_monte_carlo.py --strategy SimpleStock --tickers AAPL MSFT GOOGL --in_sample_start 2015-01-01 --in_sample_end 2019-12-31 --out_sample_start 2020-01-01 --out_sample_end 2021-12-31 --num_permutations 100
```

## Recommended Testing Workflow

For a comprehensive validation of your trading strategy, we recommend following this workflow:

1. **Parameter Optimization**: Run the In-Sample Excellence test to find optimal parameters.
   
2. **Check for Data Mining Bias**: Run the In-Sample Monte Carlo Permutation Test to ensure your strategy is capturing real patterns.
   
3. **Out-of-Sample Validation**: Run the Walk-Forward Test using the optimized parameters to check performance on unseen data.
   
4. **Statistical Significance**: Run the Walk-Forward Monte Carlo Permutation Test to validate that out-of-sample performance is statistically significant.

## Common Parameters

All test scripts share these common parameters:

- `--strategy`: Name of the strategy to test (e.g., 'SimpleStock', 'CoveredCall', 'AuctionMarket')
- `--tickers`: List of ticker symbols to test (e.g., 'AAPL MSFT GOOGL')
- `--output_dir`: Directory to save test results

## Using Optimized Parameters

You can use parameters optimized from the In-Sample Excellence test in subsequent tests:

```bash
# First, run in-sample excellence to find optimal parameters
python in_sample_excellence.py --strategy SimpleStock --output_dir output/optimization

# Then, use those parameters in the walk-forward test
python walk_forward_test.py --strategy SimpleStock --load_optimized --optimized_params_path output/optimization/best_parameters.pkl
```

## Interpreting Results

### In-Sample Excellence

- Look for high Sharpe ratios (>1.0), profit factors (>2.0), and win rates (>50%)
- Check parameter sensitivity to ensure robustness

### In-Sample Monte Carlo

- P-values < 0.05 indicate the strategy is likely capturing real patterns
- Check the distribution plots to see how your strategy compares to random

### Walk-Forward Test

- Compare in-sample and out-of-sample performance metrics
- Look for minimal degradation in key metrics like Sharpe ratio and profit factor

### Walk-Forward Monte Carlo

- P-values < 0.05 indicate the out-of-sample performance is statistically significant
- Check the distribution plots to see how your strategy compares to random permutations

## Output Files

Each test generates various output files:

- **CSV files**: Performance metrics, trade logs, and comparison tables
- **PNG files**: Equity curves, drawdowns, monthly returns, and distribution plots
- **PKL files**: Raw results data for further analysis

## Requirements

These tests require the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- backtrader
- tqdm

Install them using:

```bash
pip install -r requirements.txt
``` 