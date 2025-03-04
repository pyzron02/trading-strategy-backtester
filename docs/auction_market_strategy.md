# Auction Market Theory (AMT) Strategy Documentation

## Overview

The Auction Market Theory (AMT) strategy is a sophisticated trading approach based on the principles of auction market theory, which views markets as a continuous two-way auction process. This strategy analyzes price action, volume, and market structure to identify value areas, points of control, and imbalances in the market.

## Theoretical Background

Auction Market Theory was developed from the concepts introduced by J. Peter Steidlmayer at the Chicago Board of Trade in the 1980s. The theory posits that:

1. Markets operate as a continuous auction where buyers and sellers negotiate prices
2. Price movement is driven by the discovery of value and the resolution of imbalances
3. Trading activity tends to cluster around "fair value" areas
4. Prices move away from value areas when there is an imbalance between buyers and sellers

## Key Concepts

### Value Area

The Value Area represents the price range where a specified percentage (typically 70%) of the trading volume occurred during a given time period. This area is considered the "fair value" range where most transactions take place.

### Point of Control (POC)

The Point of Control is the price level within the Value Area where the highest volume of trading occurred. It represents the most accepted price during the time period.

### Balance and Imbalance

- **Balance**: A market condition where trading activity is distributed evenly around the Point of Control, indicating agreement on fair value.
- **Imbalance**: A market condition where trading activity shifts away from the Value Area, indicating potential directional movement.

### Excess

Excess occurs when price moves beyond the Value Area and then returns, indicating rejection of those price levels. This can signal potential reversal points.

### Rotation

Rotation refers to the movement of price from one end of the Value Area to the other, indicating a shift in market sentiment.

## Strategy Implementation

The `AuctionMarketStrategy` class implements these concepts in a systematic trading approach. The strategy is now fully contained in a single file (`auction_market_strategy.py`) which includes both the strategy implementation and the parameter management:

```python
class AuctionMarketStrategy(bt.Strategy):
    """
    Auction Market Theory trading strategy.
    
    This strategy implements concepts from Auction Market Theory including:
    - Value Area calculation (70% of volume)
    - Point of Control identification
    - Balance/Imbalance detection
    - Excess move identification
    - Rotation analysis
    
    The strategy can be configured using the AuctionMarketParameters class,
    which provides default, aggressive, and conservative parameter presets.
    """
```

## Parameter Management

The strategy uses the `AuctionMarketParameters` class for parameter management, which is now integrated into the same file:

```python
class AuctionMarketParameters:
    """Parameters for Auction Market Theory trading strategy."""
    
    def __init__(self):
        # Time-based parameters
        self.trading_hours_start = "09:30"  # Market open (EST)
        self.trading_hours_end = "16:00"    # Market close (EST)
        # ... other parameters ...
```

The file also includes functions to get predefined parameter sets:

```python
def get_default_parameters():
    """Return default parameters for Auction Market Theory strategy."""
    return AuctionMarketParameters()

def get_aggressive_parameters():
    """Return more aggressive parameters for Auction Market Theory strategy."""
    # ... parameter customization ...

def get_conservative_parameters():
    """Return more conservative parameters for Auction Market Theory strategy."""
    # ... parameter customization ...
```

## Strategy Parameters

The strategy uses the following parameters, which can be customized through the `AuctionMarketParameters` class:

### Time-Based Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `trading_hours_start` | Start of trading hours | 9:30 |
| `trading_hours_end` | End of trading hours | 16:00 |
| `lookback_days` | Number of days to look back for analysis | 10 |

### Value Area Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `value_area_volume_percent` | Percentage of volume to include in value area | 70 |
| `min_value_area_width` | Minimum width of value area as percentage of price | 1.0 |
| `poc_volume_threshold` | Minimum volume at POC relative to average | 1.5 |

### Volume Profile Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `price_levels` | Number of price levels to use in volume profile | 30 |
| `volume_smoothing` | Smoothing factor for volume profile | 2 |
| `volume_profile_days` | Days to include in volume profile | 5 |

### Trading Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `entry_threshold` | Price movement threshold for entry (% of value area) | 0.2 |
| `exit_threshold` | Price movement threshold for exit (% of value area) | 0.1 |
| `max_positions` | Maximum number of concurrent positions | 5 |
| `position_sizing` | Position sizing method ('fixed', 'percent', 'risk') | 'risk' |

### Risk Management Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `stop_loss_atr_multiple` | Stop loss as multiple of ATR | 2.0 |
| `profit_target_atr_multiple` | Profit target as multiple of ATR | 3.0 |
| `max_risk_per_trade` | Maximum risk per trade as percentage of portfolio | 1.0 |
| `trailing_stop` | Whether to use trailing stops | True |
| `trailing_stop_activation` | Profit required to activate trailing stop (% of entry) | 1.0 |

### Market Condition Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `volatility_filter` | Whether to use volatility filter | True |
| `min_atr_threshold` | Minimum ATR for trade entry | 0.5 |
| `max_atr_threshold` | Maximum ATR for trade entry | 5.0 |
| `trend_filter` | Whether to use trend filter | True |
| `trend_period` | Period for trend calculation | 20 |

### Auction Zone Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `excess_threshold` | Threshold for excess detection (% of value area) | 0.5 |
| `balance_threshold` | Threshold for balance detection (% of value area) | 0.3 |
| `rotation_threshold` | Threshold for rotation detection (% of value area) | 0.7 |

### Technical Indicators

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `use_atr` | Whether to use Average True Range | True |
| `atr_period` | Period for ATR calculation | 14 |
| `use_volume_delta` | Whether to use volume delta | True |
| `delta_period` | Period for volume delta calculation | 5 |

## Parameter Presets

The strategy includes three parameter presets:

1. **Default Parameters**: Balanced approach suitable for most market conditions
2. **Aggressive Parameters**: More sensitive to market movements, higher risk/reward
3. **Conservative Parameters**: More selective entries, lower risk/reward

## Trading Logic

The strategy's trading logic follows these steps:

1. **Data Collection**: Store daily price and volume data
2. **Value Area Calculation**: Calculate the Value Area and Point of Control
3. **Market Structure Analysis**:
   - Detect auction excess
   - Identify balance areas
   - Detect rotations
4. **Signal Generation**:
   - Enter long positions when price breaks above Value Area with confirmation
   - Enter short positions when price breaks below Value Area with confirmation
   - Exit positions when price returns to Value Area or reaches targets
5. **Position Sizing**: Calculate position size based on risk parameters
6. **Risk Management**: Apply stop losses and profit targets

## Example Signals

### Long Entry Conditions
- Price breaks above Value Area
- No excess detected above Value Area
- Rotation is upward
- Volume confirms the move

### Short Entry Conditions
- Price breaks below Value Area
- No excess detected below Value Area
- Rotation is downward
- Volume confirms the move

### Exit Conditions
- Price returns to Value Area
- Profit target reached
- Stop loss triggered
- Opposing signal generated

## Visualization

The strategy results can be visualized using the `AuctionMarketVisualizer` class, which provides:

- Equity curve plotting
- Trade distribution analysis
- P&L by symbol
- Drawdown analysis
- Monthly returns
- Trade duration vs. P&L analysis

## Running the Strategy

To run a backtest with the Auction Market Theory strategy:

```bash
python grok_code/src/engine/run_backtest.py --strategy_name AuctionMarket
```

To use specific parameter presets:

```bash
python grok_code/src/engine/run_backtest.py --strategy_name AuctionMarket --param_preset aggressive
```

## Using Custom Parameters

You can also create custom parameters and pass them to the strategy:

```python
from strategies.auction_market_strategy import AuctionMarketParameters, AuctionMarketStrategy

# Create custom parameters
params = AuctionMarketParameters()
params.value_area_volume_percent = 0.65
params.auction_zones['excess_threshold'] = 1.8

# Initialize strategy with custom parameters
strategy = AuctionMarketStrategy(parameters=params)
```

## Performance Metrics

The strategy's performance is evaluated using the following metrics:

- Total Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Average Win/Loss Ratio

## Limitations and Considerations

- The strategy is most effective in markets with sufficient volume and liquidity
- Performance may vary across different market regimes
- Parameter optimization is recommended for specific instruments
- The strategy may generate fewer signals in low-volatility environments

## Further Development

Potential areas for strategy enhancement:

- Integration with other market structure concepts
- Machine learning for parameter optimization
- Adaptive parameter adjustment based on market conditions
- Multi-timeframe analysis
- Order flow analysis integration 