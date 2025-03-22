# Trading Strategy Testing Framework

This directory contains a comprehensive set of testing tools for validating trading strategies. These tests help ensure that your strategies are robust, statistically significant, and not just fitting to historical data.

## Integrated Testing Framework

The easiest way to use this testing framework is through the `StrategyTester` class, which provides a unified interface for running all four testing methods:

```python
from engine.testing import StrategyTester

# Create the strategy tester
tester = StrategyTester(
    strategy_name='SimpleStock',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    in_sample_start='2015-01-01',
    in_sample_end='2019-12-31',
    out_sample_start='2020-01-01',
    out_sample_end='2021-12-31',
    parameter_grid={
        'sma_period': [10, 20, 50, 100, 200],
        'position_size': [10, 20, 50, 100]
    }
)

# Run all tests
results = tester.run_all_tests()
```

You can also run individual tests:

```python
# Run only the in-sample excellence test
excellence_results = tester.run_in_sample_excellence()

# Run only the walk-forward test
walk_forward_results = tester.run_walk_forward()
```

## Command Line Usage

You can also run the strategy tester from the command line:

```bash
python -m engine.testing.strategy_tester --strategy SimpleStock --tickers AAPL MSFT GOOGL --param_grid '{"sma_period": [10, 20, 50, 100, 200], "position_size": [10, 20, 50, 100]}'
```

Or use the example script for SimpleStockStrategy:

```bash
python -m engine.testing.test_simple_stock
```

## Available Tests

### 1. In-Sample Excellence

Optimizes strategy parameters on historical data to achieve strong performance metrics like high Sharpe ratios and profit factors.

```python
from engine.testing import InSampleExcellence

test = InSampleExcellence(
    strategy_name='SimpleStock',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2015-01-01',
    end_date='2019-12-31',
    parameter_grid={
        'sma_period': [10, 20, 50, 100, 200],
        'position_size': [10, 20, 50, 100]
    }
)

results = test.run_optimization()
```

### 2. In-Sample Monte Carlo Permutation Test

Compares strategy performance on real data to permuted data to check for data mining bias, ensuring results are not due to random noise.

```python
from engine.testing import InSampleMonteCarloTest

test = InSampleMonteCarloTest(
    strategy_name='SimpleStock',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2015-01-01',
    end_date='2019-12-31',
    num_permutations=100
)

results = test.run_test()
```

### 3. Walk-Forward Test

Evaluates strategy performance on out-of-sample data to assess real-world applicability.

```python
from engine.testing import WalkForwardTest

test = WalkForwardTest(
    strategy_name='SimpleStock',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    in_sample_start='2015-01-01',
    in_sample_end='2019-12-31',
    out_sample_start='2020-01-01',
    out_sample_end='2021-12-31'
)

results = test.run_test()
```

### 4. Walk-Forward Monte Carlo Permutation Test

Validates out-of-sample performance by comparing it to permuted data, ensuring statistical significance.

```python
from engine.testing import WalkForwardMonteCarloTest

test = WalkForwardMonteCarloTest(
    strategy_name='SimpleStock',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    in_sample_start='2015-01-01',
    in_sample_end='2019-12-31',
    out_sample_start='2020-01-01',
    out_sample_end='2021-12-31',
    num_permutations=100
)

results = test.run_test()
```

## Recommended Testing Workflow

For a comprehensive validation of your trading strategy, we recommend following this workflow:

1. **Parameter Optimization**: Run the In-Sample Excellence test to find optimal parameters.
   
2. **Check for Data Mining Bias**: Run the In-Sample Monte Carlo Permutation Test to ensure your strategy is capturing real patterns.
   
3. **Out-of-Sample Validation**: Run the Walk-Forward Test using the optimized parameters to check performance on unseen data.
   
4. **Statistical Significance**: Run the Walk-Forward Monte Carlo Permutation Test to validate that out-of-sample performance is statistically significant.

The `StrategyTester` class automates this entire workflow for you.

## Interpreting Results

The `StrategyTester` class generates a comprehensive summary report with interpretations of each test result and an overall recommendation. The recommendation is based on how many tests the strategy passes:

- **Strong Recommendation**: The strategy passes all four tests.
- **Positive Recommendation**: The strategy passes three out of four tests.
- **Neutral Recommendation**: The strategy passes two out of four tests.
- **Negative Recommendation**: The strategy passes only one test.
- **Strong Negative Recommendation**: The strategy fails all tests.

## Output Files

Each test generates various output files:

- **CSV files**: Performance metrics, trade logs, and comparison tables
- **PNG files**: Equity curves, drawdowns, monthly returns, and distribution plots
- **PKL files**: Raw results data for further analysis
- **TXT files**: Recommendations and interpretations

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