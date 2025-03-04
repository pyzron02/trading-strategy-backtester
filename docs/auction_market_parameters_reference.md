# Auction Market Theory Strategy - Parameter Reference Guide

This document provides a quick reference for the parameters used in the Auction Market Theory strategy.

> **Note:** The `AuctionMarketParameters` class is now integrated directly into the `auction_market_strategy.py` file. There is no longer a separate `auction_market_parameters.py` file.

## Parameter Presets

The strategy includes three built-in parameter presets that can be used out of the box:

```python
# Import from the combined file
from strategies.auction_market_strategy import get_default_parameters, get_aggressive_parameters, get_conservative_parameters

# Use default parameters
params = get_default_parameters()

# Use aggressive parameters
params = get_aggressive_parameters()

# Use conservative parameters
params = get_conservative_parameters()
```

## Parameter Categories

### Time-Based Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `trading_hours_start` | Start of trading hours | 9:30 | 9:30 | 9:30 |
| `trading_hours_end` | End of trading hours | 16:00 | 16:00 | 16:00 |
| `lookback_days` | Days to look back for analysis | 10 | 7 | 14 |

### Value Area Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `value_area_volume_percent` | % of volume in value area | 70 | 65 | 75 |
| `min_value_area_width` | Min width as % of price | 1.0 | 0.8 | 1.5 |
| `poc_volume_threshold` | Min POC volume vs avg | 1.5 | 1.3 | 1.8 |

### Volume Profile Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `price_levels` | Price levels in volume profile | 30 | 40 | 20 |
| `volume_smoothing` | Smoothing factor | 2 | 1 | 3 |
| `volume_profile_days` | Days in volume profile | 5 | 3 | 7 |

### Trading Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `entry_threshold` | Entry threshold (% of VA) | 0.2 | 0.15 | 0.3 |
| `exit_threshold` | Exit threshold (% of VA) | 0.1 | 0.05 | 0.15 |
| `max_positions` | Max concurrent positions | 5 | 7 | 3 |
| `position_sizing` | Position sizing method | 'risk' | 'risk' | 'risk' |

### Risk Management Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `stop_loss_atr_multiple` | Stop loss (ATR multiple) | 2.0 | 1.5 | 3.0 |
| `profit_target_atr_multiple` | Profit target (ATR multiple) | 3.0 | 4.0 | 2.5 |
| `max_risk_per_trade` | Max risk per trade (%) | 1.0 | 1.5 | 0.75 |
| `trailing_stop` | Use trailing stops | True | True | True |
| `trailing_stop_activation` | Trailing stop activation (%) | 1.0 | 0.75 | 1.5 |

### Market Condition Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `volatility_filter` | Use volatility filter | True | True | True |
| `min_atr_threshold` | Min ATR for entry | 0.5 | 0.3 | 0.7 |
| `max_atr_threshold` | Max ATR for entry | 5.0 | 7.0 | 4.0 |
| `trend_filter` | Use trend filter | True | False | True |
| `trend_period` | Trend calculation period | 20 | 15 | 30 |

### Auction Zone Parameters

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `excess_threshold` | Excess detection threshold | 0.5 | 0.4 | 0.6 |
| `balance_threshold` | Balance detection threshold | 0.3 | 0.2 | 0.4 |
| `rotation_threshold` | Rotation detection threshold | 0.7 | 0.6 | 0.8 |

### Technical Indicators

| Parameter | Description | Default | Aggressive | Conservative |
|-----------|-------------|---------|------------|--------------|
| `use_atr` | Use ATR | True | True | True |
| `atr_period` | ATR calculation period | 14 | 10 | 20 |
| `use_volume_delta` | Use volume delta | True | True | True |
| `delta_period` | Volume delta period | 5 | 3 | 7 |

## Command Line Usage

When running the strategy from the command line, you can specify the parameter preset:

```bash
# Run with default parameters
python grok_code/src/engine/run_backtest.py --strategy_name AuctionMarket

# Run with aggressive parameters
python grok_code/src/engine/run_backtest.py --strategy_name AuctionMarket --param_preset aggressive

# Run with conservative parameters
python grok_code/src/engine/run_backtest.py --strategy_name AuctionMarket --param_preset conservative
```

## Custom Parameters

To use custom parameters, create a new instance of `AuctionMarketParameters` and modify the desired values:

```python
from strategies.auction_market_strategy import AuctionMarketParameters, AuctionMarketStrategy

# Start with default parameters
params = AuctionMarketParameters()

# Modify specific parameters
params.value_area_volume_percent = 68
params.lookback_days = 12
params.max_positions = 4

# Use the custom parameters in the strategy
strategy = AuctionMarketStrategy(parameters=params)
```

## Parameter Optimization

For optimal results, consider optimizing these key parameters first:

1. `value_area_volume_percent`
2. `entry_threshold` and `exit_threshold`
3. `stop_loss_atr_multiple` and `profit_target_atr_multiple`
4. `excess_threshold` and `rotation_threshold`

These parameters have the most significant impact on strategy performance and can be adjusted based on the specific characteristics of the instruments being traded. 