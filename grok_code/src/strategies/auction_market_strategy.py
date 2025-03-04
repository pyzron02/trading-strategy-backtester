import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AuctionMarketParameters:
    """Parameters for Auction Market Theory trading strategy."""
    
    def __init__(self):
        # Time-based parameters
        self.trading_hours_start = "09:30"  # Market open (EST)
        self.trading_hours_end = "16:00"    # Market close (EST)
        self.profile_period = "D"           # Market profile timeframe (D=Daily)
        
        # Value Area parameters
        self.value_area_volume_percent = 0.70  # Standard 70% of volume
        self.poc_volume_threshold = 0.15       # Minimum volume for POC
        
        # Price levels and zones
        self.price_levels = {
            'tick_size': 0.01,          # Minimum price movement
            'value_area_extension': 2,   # Number of std devs for value area
            'price_bucket_size': 0.25    # Size of price buckets for distribution
        }
        
        # Volume profile parameters
        self.volume_profile = {
            'lookback_period': 20,       # Days to look back for volume profile
            'volume_threshold': 1000,    # Minimum volume for significant level
            'bucket_size': 100           # Size of volume buckets
        }
        
        # Trading parameters
        self.position_size = {
            'max_position': 100,         # Maximum position size
            'initial_size': 20,          # Initial position size
            'scaling_size': 10           # Size for scaling in/out
        }
        
        # Risk management
        self.risk_params = {
            'max_loss_percent': 0.02,    # Maximum loss per trade (2%)
            'profit_target_ratio': 2.0,  # Profit target ratio (risk:reward)
            'max_daily_loss': 0.05,      # Maximum daily loss (5%)
            'position_heat': 0.01        # Maximum heat per position (1%)
        }
        
        # Market conditions
        self.market_conditions = {
            'min_daily_volume': 1000000,  # Minimum daily volume
            'min_daily_range': 0.5,       # Minimum daily range (%)
            'max_spread': 0.05            # Maximum bid-ask spread
        }
        
        # Auction zones
        self.auction_zones = {
            'excess_threshold': 2.0,      # Standard deviations for excess
            'balance_threshold': 0.5,     # Balance area threshold
            'rotation_factor': 1.5        # Rotation detection factor
        }
        
        # Technical indicators
        self.indicators = {
            'volume_ma_period': 20,       # Volume moving average period
            'price_ma_period': 50,        # Price moving average period
            'volatility_period': 20       # Volatility calculation period
        }

def get_default_parameters():
    """Return default parameters for Auction Market Theory strategy."""
    return AuctionMarketParameters()

def get_aggressive_parameters():
    """Return more aggressive parameters for Auction Market Theory strategy."""
    params = AuctionMarketParameters()
    params.value_area_volume_percent = 0.60  # Smaller value area
    params.auction_zones['excess_threshold'] = 1.5  # Lower threshold for excess
    params.auction_zones['rotation_factor'] = 1.2  # More sensitive to rotation
    params.risk_params['max_loss_percent'] = 0.03  # Higher risk per trade
    params.risk_params['profit_target_ratio'] = 1.5  # Lower profit target
    return params

def get_conservative_parameters():
    """Return more conservative parameters for Auction Market Theory strategy."""
    params = AuctionMarketParameters()
    params.value_area_volume_percent = 0.80  # Larger value area
    params.auction_zones['excess_threshold'] = 2.5  # Higher threshold for excess
    params.auction_zones['rotation_factor'] = 2.0  # Less sensitive to rotation
    params.risk_params['max_loss_percent'] = 0.01  # Lower risk per trade
    params.risk_params['profit_target_ratio'] = 3.0  # Higher profit target
    return params

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
    
    params = (
        ('value_area_percent', 0.70),     # Standard 70% of volume for Value Area
        ('price_bucket_size', 0.25),      # Size of price buckets for distribution
        ('lookback_period', 20),          # Days to look back for volume profile
        ('excess_threshold', 2.0),        # Standard deviations for excess
        ('balance_threshold', 0.5),       # Balance area threshold
        ('rotation_factor', 1.5),         # Rotation detection factor
        ('max_position', 100),            # Maximum position size
        ('initial_size', 20),             # Initial position size
        ('scaling_size', 10),             # Size for scaling in/out
        ('max_loss_percent', 0.02),       # Maximum loss per trade (2%)
        ('profit_target_ratio', 2.0),     # Profit target ratio (risk:reward)
        ('parameters', None),             # Optional AuctionMarketParameters instance
    )
    
    def __init__(self):
        # Initialize parameters from AuctionMarketParameters if provided
        if self.p.parameters is not None:
            self._init_from_parameters(self.p.parameters)
        
        # Initialize indicators and variables
        self.value_areas = {}  # Store value areas by date
        self.poc_levels = {}   # Store Points of Control by date
        self.equity_curve = [] # For tracking equity curve
        
        # Create indicators for each data feed
        for data in self.datas:
            # Volume moving average
            data.volume_ma = bt.indicators.SimpleMovingAverage(
                data.volume, period=self.p.lookback_period
            )
            
            # Price moving averages
            data.sma50 = bt.indicators.SimpleMovingAverage(
                data.close, period=50
            )
            
            # Volatility indicator (ATR)
            data.atr = bt.indicators.ATR(
                data, period=14
            )
            
            # Store daily OHLCV for value area calculation
            data.daily_bars = []
    
    def _init_from_parameters(self, params):
        """Initialize strategy parameters from AuctionMarketParameters instance"""
        self.p.value_area_percent = params.value_area_volume_percent
        self.p.price_bucket_size = params.price_levels['price_bucket_size']
        self.p.lookback_period = params.volume_profile['lookback_period']
        self.p.excess_threshold = params.auction_zones['excess_threshold']
        self.p.balance_threshold = params.auction_zones['balance_threshold']
        self.p.rotation_factor = params.auction_zones['rotation_factor']
        self.p.max_position = params.position_size['max_position']
        self.p.initial_size = params.position_size['initial_size']
        self.p.scaling_size = params.position_size['scaling_size']
        self.p.max_loss_percent = params.risk_params['max_loss_percent']
        self.p.profit_target_ratio = params.risk_params['profit_target_ratio']
        
        print("Strategy initialized with custom parameters:")
        print(f"  Value Area: {self.p.value_area_percent}")
        print(f"  Excess Threshold: {self.p.excess_threshold}")
        print(f"  Balance Threshold: {self.p.balance_threshold}")
        print(f"  Rotation Factor: {self.p.rotation_factor}")
        print(f"  Max Loss: {self.p.max_loss_percent}")
        print(f"  Profit Target Ratio: {self.p.profit_target_ratio}")
    
    def next(self):
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
        # Process each data feed (ticker)
        for data in self.datas:
            # Store daily bar data
            self._store_daily_bar(data)
            
            # Calculate value area if we have enough data
            if len(data.daily_bars) >= 1:
                self._calculate_value_area(data)
            
            # Get current value area if available
            current_date = data.datetime.date(0)
            value_area = self.value_areas.get(current_date, None)
            
            # Trading logic
            if value_area:
                self._apply_auction_market_logic(data, value_area)
    
    def _store_daily_bar(self, data):
        """Store daily bar data for value area calculation"""
        current_date = data.datetime.date(0)
        
        # Create a new daily bar entry
        bar = {
            'date': current_date,
            'open': data.open[0],
            'high': data.high[0],
            'low': data.low[0],
            'close': data.close[0],
            'volume': data.volume[0]
        }
        
        # Add to daily bars list
        data.daily_bars.append(bar)
        
        # Keep only the lookback period
        if len(data.daily_bars) > self.p.lookback_period:
            data.daily_bars.pop(0)
    
    def _calculate_value_area(self, data):
        """Calculate Value Area and Point of Control"""
        current_date = data.datetime.date(0)
        
        # Get the most recent daily bar
        daily_bar = data.daily_bars[-1]
        
        # Create price buckets
        price_range = np.arange(
            daily_bar['low'],
            daily_bar['high'] + self.p.price_bucket_size,
            self.p.price_bucket_size
        )
        
        # Calculate volume distribution
        volume_dist = {}
        for price in price_range:
            # Simple approximation: distribute volume across price range
            if price >= daily_bar['low'] and price <= daily_bar['high']:
                # Weight volume more heavily near the close price
                weight = 1.0 - abs(price - daily_bar['close']) / (daily_bar['high'] - daily_bar['low'])
                volume_dist[price] = daily_bar['volume'] * max(0.1, weight)
            else:
                volume_dist[price] = 0
        
        # Find POC (price with highest volume)
        poc = max(volume_dist.items(), key=lambda x: x[1])[0] if volume_dist else daily_bar['close']
        
        # Calculate Value Area
        total_volume = sum(volume_dist.values())
        target_volume = total_volume * self.p.value_area_percent
        current_volume = volume_dist.get(poc, 0)
        
        vah = poc  # Value Area High
        val = poc  # Value Area Low
        
        # Expand value area until it contains target volume
        price_list = sorted(price_range)
        poc_idx = price_list.index(poc) if poc in price_list else len(price_list) // 2
        
        above_idx = poc_idx
        below_idx = poc_idx
        
        while current_volume < target_volume and (above_idx < len(price_list) - 1 or below_idx > 0):
            # Look for next prices above and below
            above_price = price_list[above_idx + 1] if above_idx < len(price_list) - 1 else None
            below_price = price_list[below_idx - 1] if below_idx > 0 else None
            
            # Get volumes
            above_vol = volume_dist.get(above_price, 0) if above_price else 0
            below_vol = volume_dist.get(below_price, 0) if below_price else 0
            
            # Add the larger volume to the value area
            if above_vol > below_vol and above_price:
                above_idx += 1
                vah = above_price
                current_volume += above_vol
            elif below_price:
                below_idx -= 1
                val = below_price
                current_volume += below_vol
            else:
                break
        
        # Store value area
        self.value_areas[current_date] = {
            'poc': poc,
            'vah': vah,
            'val': val,
            'volume_profile': volume_dist
        }
        
        # Store POC
        self.poc_levels[current_date] = poc
        
        print(f"{current_date} - {data._name}: Value Area: {val:.2f} - {vah:.2f}, POC: {poc:.2f}")
    
    def _detect_auction_excess(self, data):
        """Detect excess moves beyond the value area"""
        current_date = data.datetime.date(0)
        value_area = self.value_areas.get(current_date, None)
        
        if not value_area:
            return {'above': False, 'below': False}
        
        # Calculate price volatility
        price_std = data.atr[0]
        
        # Check for excess above value area
        above_excess = data.high[0] > value_area['vah'] + (price_std * self.p.excess_threshold)
        
        # Check for excess below value area
        below_excess = data.low[0] < value_area['val'] - (price_std * self.p.excess_threshold)
        
        return {
            'above': above_excess,
            'below': below_excess
        }
    
    def _identify_balance_area(self, data):
        """Identify balanced trading ranges"""
        # Calculate price movement statistics
        price_range = data.high[0] - data.low[0]
        
        # Use ATR as a measure of average range
        avg_range = data.atr[0]
        
        # Check for balanced conditions
        is_balanced = price_range < avg_range * self.p.balance_threshold
        
        return {
            'is_balanced': is_balanced,
            'balance_high': data.high[0] if is_balanced else None,
            'balance_low': data.low[0] if is_balanced else None
        }
    
    def _detect_rotation(self, data):
        """Detect price rotation between value areas"""
        current_date = data.datetime.date(0)
        value_area = self.value_areas.get(current_date, None)
        
        if not value_area:
            return {'detected': False, 'direction': None}
        
        # Calculate price movement
        price_change = data.close[0] - data.close[-1]
        
        # Calculate rotation factor
        value_area_size = value_area['vah'] - value_area['val']
        
        if value_area_size > 0 and abs(price_change) > value_area_size * self.p.rotation_factor:
            return {
                'detected': True,
                'direction': 'up' if price_change > 0 else 'down'
            }
        
        return {'detected': False, 'direction': None}
    
    def _calculate_position_size(self, data, risk_level):
        """Calculate position size based on risk parameters"""
        account_size = self.broker.getvalue()
        risk_amount = account_size * self.p.max_loss_percent
        
        # Ensure risk level is not zero
        risk_level = max(0.001, risk_level)
        
        position_size = min(
            self.p.max_position,
            int(risk_amount / (data.close[0] * risk_level))
        )
        
        return max(1, position_size)  # Ensure at least 1 share
    
    def _apply_auction_market_logic(self, data, value_area):
        """Apply Auction Market Theory trading logic"""
        current_date = data.datetime.date(0)
        current_price = data.close[0]
        position = self.getposition(data).size
        
        # Get market conditions
        excess = self._detect_auction_excess(data)
        balance = self._identify_balance_area(data)
        rotation = self._detect_rotation(data)
        
        print(f"{current_date} - {data._name}: Price: {current_price:.2f}, Position: {position}")
        print(f"  Value Area: {value_area['val']:.2f} - {value_area['vah']:.2f}, POC: {value_area['poc']:.2f}")
        print(f"  Excess: Above={excess['above']}, Below={excess['below']}")
        print(f"  Balance: {balance['is_balanced']}")
        print(f"  Rotation: {rotation['detected']} {rotation['direction'] if rotation['detected'] else ''}")
        
        # Long signal conditions
        if (current_price < value_area['val'] and 
            not excess['below'] and 
            not balance['is_balanced'] and
            position <= 0):
            
            risk_level = (value_area['val'] - current_price) / current_price
            position_size = self._calculate_position_size(data, risk_level)
            
            self.buy(data=data, size=position_size)
            print(f"{current_date} - BUY signal for {data._name}: Size={position_size}, Price={current_price:.2f}")
            print(f"  Reason: Price below value area without excess")
        
        # Short signal conditions
        elif (current_price > value_area['vah'] and 
              not excess['above'] and 
              not balance['is_balanced'] and
              position >= 0):
            
            risk_level = (current_price - value_area['vah']) / current_price
            position_size = self._calculate_position_size(data, risk_level)
            
            self.sell(data=data, size=position_size)
            print(f"{current_date} - SELL signal for {data._name}: Size={position_size}, Price={current_price:.2f}")
            print(f"  Reason: Price above value area without excess")
        
        # Exit long position
        elif position > 0 and (
            current_price > value_area['poc'] or  # Price above POC
            excess['below'] or                    # Excess below value area
            rotation['detected'] and rotation['direction'] == 'down'  # Downward rotation
        ):
            self.close(data=data)
            print(f"{current_date} - CLOSE LONG position for {data._name}: Price={current_price:.2f}")
        
        # Exit short position
        elif position < 0 and (
            current_price < value_area['poc'] or  # Price below POC
            excess['above'] or                    # Excess above value area
            rotation['detected'] and rotation['direction'] == 'up'  # Upward rotation
        ):
            self.close(data=data)
            print(f"{current_date} - CLOSE SHORT position for {data._name}: Price={current_price:.2f}")
    
    def notify_order(self, order):
        """Log order execution information"""
        if order.status in [order.Completed]:
            if order.isbuy():
                action = "BUY"
            else:
                action = "SELL"
            
            print(f"{self.data.datetime.date(0)} - {action} order executed: "
                  f"Price={order.executed.price:.2f}, Size={order.executed.size}, "
                  f"Value=${order.executed.value:.2f}, Commission=${order.executed.comm:.2f}") 