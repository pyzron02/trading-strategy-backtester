# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running workflows
- Basic backtest: `python src/utils/run_simple_workflow.py --strategy [StrategyName] --tickers [Tickers]`
- Monte Carlo: `python run_trade_monte_carlo.py --strategy [StrategyName] --tickers [Tickers] --num-simulations [N]` 
- CLI interface: `python src/workflows/cli.py --workflow [type] --strategy [StrategyName] --tickers [Tickers]`
- Using config file: `python src/workflows/cli.py --config [config_file.json]`
- Unified workflow: `python src/workflows/unified_workflow.py [config_file.json]`
- Testing strategies: `python src/evaluators/test_strategies.py`

## Code Style Guidelines

- **Imports**: Group standard library first, then third-party, then local imports
- **Docstrings**: Use Google style docstrings with Args/Returns sections
- **Error handling**: Use try/except with specific exception types and meaningful error messages
- **Type hints**: Optional but encouraged for function parameters and return values
- **Naming**: Use snake_case for functions/variables, PascalCase for classes, follow Strategy naming convention
- **Logging**: Use the logging_system module for consistent logging across the codebase
- **Parameters**: Strategy parameters should be passed as dictionaries
- **Testing**: Use the StrategyTester class for strategy validation