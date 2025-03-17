# Strategy Testing Examples

This directory contains example scripts that demonstrate how to use the trading strategy testing framework.

## Available Examples

### 1. SimpleStock Strategy Testing

The `test_simple_stock_example.py` script demonstrates how to use the integrated testing framework with the SimpleStock strategy.

```bash
python -m examples.test_simple_stock_example
```

This example:
- Runs all four tests on the SimpleStock strategy
- Demonstrates how to configure the parameter grid for optimization
- Shows how to run individual tests
- Prints a comprehensive summary report with recommendations

## Creating Your Own Tests

You can use these examples as templates for testing your own strategies. Here's a basic workflow:

1. **Create a Strategy**: Implement your strategy in the `strategies` directory.

2. **Define Parameters**: Identify the parameters you want to optimize.

3. **Configure Testing**: Create a script similar to the examples that configures the StrategyTester with your strategy and parameters.

4. **Run Tests**: Execute the script to run the tests.

5. **Analyze Results**: Review the summary report and recommendations to determine if your strategy is robust and statistically significant.

## Command Line Arguments

Most example scripts support command line arguments for customization:

```bash
python -m examples.test_simple_stock_example --tickers AAPL MSFT GOOGL --in_sample_start 2015-01-01 --in_sample_end 2019-12-31
```

Run the script with `--help` to see all available options:

```bash
python -m examples.test_simple_stock_example --help
```

## Output Files

Each test generates various output files in the specified output directory:

- **CSV files**: Performance metrics, trade logs, and comparison tables
- **PNG files**: Equity curves, drawdowns, monthly returns, and distribution plots
- **PKL files**: Raw results data for further analysis
- **TXT files**: Recommendations and interpretations

The summary report provides a comprehensive overview of all test results and an overall recommendation for the strategy. 