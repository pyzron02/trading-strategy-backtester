#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auction Market Theory Trading Strategy.

This strategy implements concepts from Auction Market Theory including
Value Area calculation, Point of Control identification, and more.
"""

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
        ('param_preset', 'default'),  # Parameter preset (default, aggressive, conservative)
        ('value_area', 0.7),          # Value Area (percentage of volume)
        ('use_vwap', True),           # Use VWAP in analysis
        ('use_volume_profile', True), # Use volume profile analysis
        ('position_size', 100),       # Default position size
        ('risk_percent', 0.01),       # Risk 1% per trade by default
        ('use_atr_sizing', True),     # Use ATR for position sizing
        ('atr_period', 14),           # ATR calculation period
    )

    def __init__(self):
        """Initialize the strategy with Auction Market Theory indicators."""
        # Convert potentially float parameters to appropriate types
        # This is needed when parameters come from optimization with parameter grid
        self.p.position_size = int(self.p.position_size) if isinstance(self.p.position_size, (int, float)) else self.p.position_size
        self.p.atr_period = int(self.p.atr_period) if isinstance(self.p.atr_period, (int, float)) else self.p.atr_period
        
        # Handle list parameters that could come from optimization
        if isinstance(self.p.position_size, list):
            self.p.position_size = int(self.p.position_size[0])
        if isinstance(self.p.atr_period, list):
            self.p.atr_period = int(self.p.atr_period[0])
        if isinstance(self.p.value_area, list):
            self.p.value_area = float(self.p.value_area[0])
        if isinstance(self.p.risk_percent, list):
            self.p.risk_percent = float(self.p.risk_percent[0])
            
        # Initialize parameters from AuctionMarketParameters if provided
        if self.params.param_preset == 'default':
            self.amt_params = get_default_parameters()
        elif self.params.param_preset == 'aggressive':
            self.amt_params = get_aggressive_parameters()
        elif self.params.param_preset == 'conservative':
            self.amt_params = get_conservative_parameters()
        else:
            self.amt_params = AuctionMarketParameters()
        
        self._init_from_parameters(self.amt_params)
        
        # Create storage for daily bars and value areas
        self.daily_bars = {}
        self.value_areas = {}
        
        # Set minimum periods to ensure indicators have enough data
        # Use the largest required period times a safety factor
        min_period = max(self.params.atr_period, 50) * 3
        self.addminperiod(min_period)
        
        # For position sizing - use dictionary-based indicator storage for safety
        self.atr = {}
        self.volume_ma = {}
        self.sma50 = {}
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Keep track of trades for analysis
        self.trades = []
        
        # Initialize indicators and variables
        self.value_areas = {}  # Store value areas by date
        self.poc_levels = {}   # Store Points of Control by date
        
        # Variables to track if we have enough data for trading
        self.bars_processed = 0
        self.min_bars_required = min_period  # Ensure sufficient warmup
        
        # Create indicators for each data feed
        for data in self.datas:
            # Volume moving average
            self.volume_ma[data] = bt.indicators.SimpleMovingAverage(
                data.volume, period=self.amt_params.volume_profile['lookback_period']
            )
            
            # Price moving averages
            self.sma50[data] = bt.indicators.SimpleMovingAverage(
                data.close, period=50
            )
            
            # Volatility indicator (ATR)
            self.atr[data] = bt.indicators.ATR(
                data, period=self.params.atr_period
            )
            
            # Store daily OHLCV for value area calculation
            self.daily_bars[data] = []
    
    def _init_from_parameters(self, params):
        """Initialize strategy parameters from AuctionMarketParameters instance"""
        self.amt_params.value_area_percent = params.value_area_volume_percent
        self.amt_params.price_levels['price_bucket_size'] = params.price_levels['price_bucket_size']
        self.amt_params.volume_profile['lookback_period'] = params.volume_profile['lookback_period']
        self.amt_params.auction_zones['excess_threshold'] = params.auction_zones['excess_threshold']
        
        # Ensure balance_threshold is not zero to avoid division by zero
        if params.auction_zones['balance_threshold'] <= 0:
            print("Warning: balance_threshold cannot be zero or negative. Setting to default 0.5")
            self.amt_params.auction_zones['balance_threshold'] = 0.5
        else:
            self.amt_params.auction_zones['balance_threshold'] = params.auction_zones['balance_threshold']
            
        self.amt_params.auction_zones['rotation_factor'] = params.auction_zones['rotation_factor']
        self.amt_params.position_size['max_position'] = params.position_size['max_position']
        self.amt_params.position_size['initial_size'] = params.position_size['initial_size']
        self.amt_params.position_size['scaling_size'] = params.position_size['scaling_size']
        self.amt_params.risk_params['max_loss_percent'] = params.risk_params['max_loss_percent']
        self.amt_params.risk_params['profit_target_ratio'] = params.risk_params['profit_target_ratio']
        
        print("Strategy initialized with custom parameters:")
        print(f"  Value Area: {self.amt_params.value_area_percent}")
        print(f"  Excess Threshold: {self.amt_params.auction_zones['excess_threshold']}")
        print(f"  Balance Threshold: {self.amt_params.auction_zones['balance_threshold']}")
        print(f"  Rotation Factor: {self.amt_params.auction_zones['rotation_factor']}")
        print(f"  Max Loss: {self.amt_params.risk_params['max_loss_percent']}")
        print(f"  Profit Target Ratio: {self.amt_params.risk_params['profit_target_ratio']}")
    
    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        try:
            date = self.data.datetime.date(0).isoformat()
            value = self.broker.getvalue()
            self.equity_curve.append({'Date': date, 'Value': value})
        except Exception as e:
            print(f"Error logging portfolio value: {e}")
        
        # Skip trading until we have enough bars for indicators to be reliable
        if self.bars_processed < self.min_bars_required:
            return
            
        # Process each data feed
        for data in self.datas:
            try:
                # Safety check to ensure data feed has enough bars
                if len(data) < self.min_bars_required:
                    continue
                
                # Safety check to ensure all indicators have valid values
                if (not self.atr.get(data) or len(self.atr[data]) == 0 or not self.atr[data][0] or 
                    not self.volume_ma.get(data) or len(self.volume_ma[data]) == 0 or not self.volume_ma[data][0] or
                    not self.sma50.get(data) or len(self.sma50[data]) == 0 or not self.sma50[data][0]):
                    continue
                    
                # Store the daily bar for value area calculation
                self._store_daily_bar(data)
            
                # Only proceed if we have enough daily bars
                if not self.daily_bars.get(data) or len(self.daily_bars[data]) < self.amt_params.volume_profile['lookback_period']:
                    continue
                
                # Calculate value area if available
                try:
                    value_area = self._calculate_value_area(data)
                    if not value_area:
                        continue
                    
                    # Apply auction market logic
                    self._apply_auction_market_logic(data, value_area)
                except Exception as e:
                    print(f"Error calculating value area or applying trading logic: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
            except Exception as e:
                print(f"Error in next() for {data._name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _store_daily_bar(self, data):
        """Store daily bar data for value area calculation"""
        try:
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
            if data not in self.daily_bars:
                self.daily_bars[data] = []
            
            self.daily_bars[data].append(bar)
        
            # Keep only the lookback period
            if len(self.daily_bars[data]) > self.amt_params.volume_profile['lookback_period']:
                self.daily_bars[data].pop(0)
        except Exception as e:
            print(f"Error storing daily bar: {e}")
    
    def _calculate_value_area(self, data):
        """Calculate Value Area and Point of Control"""
        try:
            current_date = data.datetime.date(0)
        
            # Get the most recent daily bar
            if not self.daily_bars.get(data) or len(self.daily_bars[data]) == 0:
                return None
                
            daily_bar = self.daily_bars[data][-1]
            
            # Safety check for price range
            if daily_bar['high'] <= daily_bar['low'] or daily_bar['high'] - daily_bar['low'] < 0.0001:
                # Invalid price range, skip
                return None
        
            # Create price buckets
            try:
                price_range = np.arange(
                    daily_bar['low'],
                    daily_bar['high'] + self.amt_params.price_levels['price_bucket_size'],
                    self.amt_params.price_levels['price_bucket_size']
                )
            except Exception as e:
                print(f"Error creating price range: {e}")
                # Fallback to a simple price range
                price_range = np.linspace(daily_bar['low'], daily_bar['high'], 20)
        
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
            if not volume_dist:
                return None
                
            poc = max(volume_dist.items(), key=lambda x: x[1])[0] if volume_dist else daily_bar['close']
        
            # Calculate Value Area
            total_volume = sum(volume_dist.values())
            if total_volume <= 0:
                return None
                
            target_volume = total_volume * self.amt_params.value_area_percent
            current_volume = volume_dist.get(poc, 0)
        
            vah = poc  # Value Area High
            val = poc  # Value Area Low
        
            # Expand value area until it contains target volume
            price_list = sorted(price_range)
            if not price_list:
                return None
                
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
            if current_date not in self.value_areas:
                self.value_areas[current_date] = {}
                
            self.value_areas[current_date] = {
                'poc': poc,
                'vah': vah,
                'val': val,
                'volume_profile': volume_dist
            }
        
            # Store POC
            self.poc_levels[current_date] = poc
        
            print(f"{current_date} - {data._name}: Value Area: {val:.2f} - {vah:.2f}, POC: {poc:.2f}")
            
            return self.value_areas[current_date]
        except Exception as e:
            print(f"Error calculating value area: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_auction_excess(self, data):
        """Detect price excesses outside the value area."""
        try:
            # Get current value area
            current_date = data.datetime.date(0)
            value_area = self.value_areas.get(current_date, None)
        
            if not value_area:
                return None
            
            # Safety check for required fields
            if 'vah' not in value_area or 'val' not in value_area:
                return None
            
            # Calculate price volatility with safety check
            if not self.atr.get(data) or not self.atr[data][0]:
                return None
                
            price_std = self.atr[data][0]
        
            # Check for excess above value area
            if data.high[0] > value_area['vah'] + (price_std * self.amt_params.auction_zones['excess_threshold']):
                return "up"
        
            # Check for excess below value area
            if data.low[0] < value_area['val'] - (price_std * self.amt_params.auction_zones['excess_threshold']):
                return "down"
            
            return None
        except Exception as e:
            print(f"Error detecting auction excess: {e}")
            return None
    
    def _identify_balance_area(self, data):
        """Identify balanced vs. imbalanced market conditions."""
        try:
            # Safety check for atr
            if not self.atr.get(data) or not self.atr[data][0]:
                return "normal"  # Default to normal if ATR not available
        
        # Use ATR as a measure of average range
            avg_range = self.atr[data][0]
            
            # Ensure high and low data is available
            if not data.high or not data.low:
                return "normal"
        
        # Check for balanced conditions
            range_today = data.high[0] - data.low[0]
            
            # Check if balance_threshold is not zero to avoid division by zero
            if self.amt_params.auction_zones['balance_threshold'] <= 0:
                return "normal"  # Safety fallback

            if range_today < avg_range * self.amt_params.auction_zones['balance_threshold']:
                return "tight"
            elif range_today > avg_range * (2.0 / self.amt_params.auction_zones['balance_threshold']):
                return "wide"
            else:
                return "normal"
        except Exception as e:
            print(f"Error identifying balance area: {e}")
            return "normal"  # Default to normal on error
    
    def _detect_rotation(self, data):
        """Detect rotations in the market by analyzing price movement patterns."""
        try:
            # Safety checks for required data
            if not data.close or len(data.close) < 2:
                return None
                
            if not self.sma50.get(data) or not self.sma50[data][0]:
                return None
            
            # Detect price rotation using ATR and moving averages
            current_close = data.close[0]
            current_open = data.open[0]
            
            # Check for rotation from below value area to above
            current_ma = self.sma50[data][0]  # Use 50-period SMA as trend reference
            
            if data.close[-1] < current_ma and current_close > current_ma:
                return "up"
            elif data.close[-1] > current_ma and current_close < current_ma:
                return "down"
            else:
                return None
        except Exception as e:
            print(f"Error detecting rotation: {e}")
            return None
    
    def _calculate_position_size(self, data, risk_level):
        """Calculate position size based on risk parameters."""
        try:
            # Base position size on account equity and volatility
            portfolio_value = self.broker.getvalue()
            risk_amount = portfolio_value * self.amt_params.risk_params['max_loss_percent']
            
            # Safety check for ATR
            if not self.atr.get(data) or not self.atr[data][0]:
                return self.amt_params.position_size['initial_size']
            
            # Use ATR for volatility-based position sizing
            if self.params.use_atr_sizing and self.atr[data][0]:
                # Calculate risk per share based on ATR
                risk_per_share = self.atr[data][0] * risk_level
                # Ensure risk_per_share is not zero to avoid division by zero
                if risk_per_share <= 0:
                    print(f"Warning: Invalid risk_per_share value ({risk_per_share}). Using default position size.")
                    return self.amt_params.position_size['initial_size']
                    
                pos_size = int(risk_amount / risk_per_share)
                # Cap position size for safety
                max_size = self.amt_params.position_size['max_position']
                return max(1, min(pos_size, max_size))
            else:
                # Use fixed position size from parameters
                return self.amt_params.position_size['initial_size']
        except Exception as e:
            print(f"Error calculating position size: {e}, using default")
            return 10  # Default safe position size on error
    
    def _apply_auction_market_logic(self, data, value_area):
        """Apply Auction Market Theory trading logic."""
        try:
            # Safety checks for required data
            if not value_area or not data.close or not data.high or not data.low:
                return
                
            if 'vah' not in value_area or 'val' not in value_area or 'poc' not in value_area:
                return
            
            # Current position
            position = self.getposition(data).size
            
            # Current price and value area
            close = data.close[0]
            vah = value_area['vah']  # Value area high
            val = value_area['val']  # Value area low
            poc = value_area['poc']  # Point of control
            
            # Check for excess moves with safety check
            excess = self._detect_auction_excess(data)
            
            # Check for balance/imbalance with safety check
            balance = self._identify_balance_area(data)
            
            # Check for rotation with safety check
            rotation = self._detect_rotation(data)
            
            # Calculate appropriate position size with safety check
            risk_level = 1.0  # Standard risk level
            if excess:
                # Reduce risk if in excess area
                risk_level = 0.5
            
            pos_size = self._calculate_position_size(data, risk_level)
            
            # Safety check for indicator values
            if not self.sma50.get(data) or not self.sma50[data][0] or not self.volume_ma.get(data) or not self.volume_ma[data][0]:
                return
            
            # Trading logic based on auction market principles
            if position == 0:  # No position
                if close > vah:  # Price above value area high
                    if excess == "up":
                        # Excess above value area - potential reversal
                        if balance == "tight" and rotation == "down":
                            # Short when we see excess up, tight balance, and downward rotation
                            self.sell(data=data, size=pos_size)
                    else:
                        # No excess - potential breakout
                        if rotation == "up" and close > self.sma50[data][0]:
                            # Go long above value area with upward rotation and above MA
                            self.buy(data=data, size=pos_size)
                
                elif close < val:  # Price below value area low
                    if excess == "down":
                        # Excess below value area - potential reversal
                        if balance == "tight" and rotation == "up":
                            # Go long when we see excess down, tight balance, and upward rotation
                            self.buy(data=data, size=pos_size)
                else:  # Price inside value area
                    if close > poc and rotation == "up" and self.volume_ma[data][0] < data.volume[0]:
                        # Go long above POC with upward rotation and above-average volume
                        self.buy(data=data, size=int(pos_size * 0.7))  # Reduced size in value area
                    elif close < poc and rotation == "down" and self.volume_ma[data][0] < data.volume[0]:
                        # Go short below POC with downward rotation and above-average volume
                        self.sell(data=data, size=int(pos_size * 0.7))  # Reduced size in value area
            
            elif position > 0:  # Long position
                if (close < val and balance != "wide") or close < (val - self.atr[data][0]):
                    # Exit long if price drops below value area low or too far below
                    self.close(data=data)
                elif excess == "up" and balance == "tight":
                    # Take partial profits on excess above value area
                    self.sell(data=data, size=int(position * 0.5))
            
            elif position < 0:  # Short position
                if (close > vah and balance != "wide") or close > (vah + self.atr[data][0]):
                    # Exit short if price rises above value area high or too far above
                    self.close(data=data)
                elif excess == "down" and balance == "tight":
                    # Take partial profits on excess below value area
                    self.buy(data=data, size=int(abs(position) * 0.5))
        except Exception as e:
            print(f"Error applying auction market logic: {e}")
            import traceback
            traceback.print_exc()
    
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