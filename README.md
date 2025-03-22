# Trading Strategy Backtester

A comprehensive backtesting framework for trading strategies, with support for optimization, walk-forward testing, and Monte Carlo simulations.

## Features

- Backtesting of various trading strategies
- Parameter optimization
- Walk-forward testing
- Monte Carlo simulation
- Multi-CPU support for enhanced performance
- Extensive trade analysis and reporting

## Documentation

All documentation has been consolidated in the `docs` directory:

- [Main Documentation](docs/main_readme.md) - Complete documentation of the framework
- [Monte Carlo Testing](docs/README_MONTE_CARLO.md) - Guide to Monte Carlo testing
- [Monte Carlo Changes](docs/CHANGES_MONTE_CARLO.md) - Changes related to Monte Carlo implementation
- [Serialization](docs/README_SERIALIZATION.md) - Handling serialization in the framework

### Strategy Documentation

- [Auction Market Strategy](docs/auction_market_strategy.md) - Documentation for the Auction Market strategy
- [Auction Market Parameters](docs/auction_market_parameters_reference.md) - Reference for Auction Market parameters
- [Analyzing AMT Results](docs/analyzing_amt_results.md) - Guide to analyzing Auction Market test results

### Source Code Documentation

- [Source Code Overview](docs/src/README.md) - Overview of the source code structure
- [Streamlining](docs/src/STREAMLINING.md) - Code streamlining documentation
- [Engine Testing](docs/src/engine_testing.md) - Documentation for the testing engine
- [Examples](docs/src/examples.md) - Examples of using the framework

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# Run a simple backtest
python -m src.workflows.unified_workflow --workflow-type simple --strategy SimpleStock --tickers AAPL --start-date 2020-01-01 --end-date 2021-01-01

# Run a complete workflow with optimization and Monte Carlo testing
python -m src.workflows.unified_workflow --workflow-type complete --strategy SimpleStock --tickers AAPL --start-date 2020-01-01 --end-date 2021-01-01 --num-cores 4
```

## License

GPL-3.0 