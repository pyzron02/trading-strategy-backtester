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
            'max_loss_percent': 0.01,    # Maximum loss per trade (1% - balanced)
            'profit_target_ratio': 2.0,  # Profit target ratio (risk:reward)
            'max_daily_loss': 0.03,      # Maximum daily loss (3% - reasonable)
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
        self.last_trade_bar = 0  # Track when last trade occurred
        
        # Initialize indicators and variables
        self.value_areas = {}  # Store value areas by date
        self.poc_levels = {}   # Store Points of Control by date
        
        # Variables to track if we have enough data for trading
        self.bars_processed = 0
        self.min_bars_required = min_period  # Ensure sufficient warmup
        
        # Parameter relaxation settings
        self.param_relaxation_enabled = True
        self.bars_without_trade = 0
        self.relaxation_factor = 1.0  # Multiplier for entry conditions
        self.max_relaxation = 0.5  # Maximum relaxation (50% of original)
        
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
        
        # Update parameter relaxation if enabled
        if self.param_relaxation_enabled:
            self._update_parameter_relaxation()
        
        # Log portfolio value for the equity curve
        try:
            date = self.data.datetime.date(0).isoformat()
            value = self.broker.getvalue()
            
            # Detect and handle holiday/missing data issues
            if len(self.equity_curve) > 0:
                prev_value = self.equity_curve[-1]['Value']
                # If portfolio value drops or spikes by more than 5% in one day, it's likely a data issue
                pct_change = abs((value - prev_value) / prev_value) if prev_value != 0 else 0
                
                # Check if current bar has valid price data
                has_valid_prices = True
                for data in self.datas:
                    if (not data.close[0] or data.close[0] <= 0 or 
                        not data.open[0] or data.open[0] <= 0 or
                        not data.high[0] or data.high[0] <= 0 or
                        not data.low[0] or data.low[0] <= 0):
                        has_valid_prices = False
                        break
                
                if pct_change > 0.05 and not has_valid_prices:
                    # This is likely a holiday/data issue - use previous value
                    print(f"WARNING: Detected invalid portfolio value on {date} (change: {pct_change:.1%}). Using previous value.")
                    value = prev_value
            
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
                try:
                    if (not self.atr.get(data) or len(self.atr[data]) == 0 or 
                        not self.volume_ma.get(data) or len(self.volume_ma[data]) == 0 or
                        not self.sma50.get(data) or len(self.sma50[data]) == 0):
                        continue
                        
                    # Check values are valid (not NaN, not zero for ATR)
                    if (not self.atr[data][0] or self.atr[data][0] <= 0 or
                        not self.volume_ma[data][0] or self.volume_ma[data][0] <= 0 or  
                        not self.sma50[data][0] or self.sma50[data][0] <= 0):
                        continue
                except (IndexError, TypeError, KeyError):
                    continue
                
                # Validate price data before trading
                if not self._is_valid_price_data(data):
                    current_date = data.datetime.date(0)
                    print(f"WARNING: Skipping trading on {current_date} due to invalid price data")
                    # Cancel any pending orders to prevent execution on invalid price data
                    self._cancel_pending_orders()
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
            
            # Get relaxed threshold
            excess_threshold = self._get_relaxed_threshold(self.amt_params.auction_zones['excess_threshold'])
        
            # Check for excess above value area
            if data.high[0] > value_area['vah'] + (price_std * excess_threshold):
                return "up"
        
            # Check for excess below value area
            if data.low[0] < value_area['val'] - (price_std * excess_threshold):
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
            
            # Get relaxed threshold
            balance_threshold = self._get_relaxed_threshold(self.amt_params.auction_zones['balance_threshold'])

            if range_today < avg_range * balance_threshold:
                return "tight"
            elif range_today > avg_range * (2.0 / balance_threshold):
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
        """Calculate position size based on risk parameters with improved safety."""
        try:
            # Get available cash and current position value
            portfolio_value = self.broker.getvalue()
            current_position_value = abs(self.broker.getposition(data).size * data.close[0])
            available_cash = self.broker.getcash()
            current_price = data.close[0]
            
            # Calculate maximum position value (percentage of available cash, not total portfolio)
            max_position_value = min(available_cash * 0.8, portfolio_value * 0.15)  # Max 15% of total portfolio or 80% of cash
            
            # Calculate risk-based position sizing using available cash, not total portfolio
            risk_amount = available_cash * self.amt_params.risk_params['max_loss_percent']
            
            # Safety check for ATR - handle array index errors
            atr_value = None
            try:
                if self.atr.get(data) and len(self.atr[data]) > 0:
                    atr_value = self.atr[data][0]
            except (IndexError, TypeError):
                atr_value = None
                
            if not atr_value or atr_value <= 0:
                # Use conservative fixed position size when ATR unavailable
                pos_size = min(20, int(max_position_value / current_price))
                return max(1, pos_size)
            
            # Use ATR for volatility-based position sizing
            if self.params.use_atr_sizing and atr_value:
                # Use a more conservative risk calculation
                
                # Calculate stop loss distance (2 * ATR for breathing room)
                stop_distance = atr_value * 2.0 * risk_level
                
                # Ensure stop distance is reasonable (not too small or large)
                min_stop = current_price * 0.02  # Minimum 2% stop
                max_stop = current_price * 0.10  # Maximum 10% stop
                stop_distance = max(min_stop, min(stop_distance, max_stop))
                
                # Calculate position size based on risk amount and stop distance
                pos_size = int(risk_amount / stop_distance)
                
                # Apply multiple safety caps
                max_size_risk = self.amt_params.position_size['max_position']
                max_size_value = int(max_position_value / current_price)
                max_size_conservative = min(100, max_size_risk)  # Conservative max 100 shares
                max_size_cash = int(available_cash * 0.5 / current_price)  # Never use more than 50% of cash
                
                final_size = max(1, min(pos_size, max_size_risk, max_size_value, max_size_conservative, max_size_cash))
                
                # Log position sizing for debugging
                if final_size != pos_size:
                    print(f"Position size capped: calculated={pos_size}, final={final_size} "
                          f"(price=${current_price:.2f}, cash=${available_cash:.0f}, ATR={atr_value:.2f}, stop=${stop_distance:.2f})")
                
                return final_size
            else:
                # Use fixed position size with value cap
                fixed_size = self.amt_params.position_size['initial_size']
                max_size_value = int(max_position_value / current_price)
                return max(1, min(fixed_size, max_size_value))
                
        except Exception as e:
            print(f"Error calculating position size: {e}, using safe default")
            return 5  # Very conservative default on error
    
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
            
            # Simplified trading logic with fewer conditions
            if position == 0:  # No position
                entry_signal = False
                entry_size = pos_size
                entry_action = None
                
                # Score-based entry system (accumulate signal strength)
                entry_score = 0
                
                # Value area signals (primary)
                if close > vah:
                    entry_score += 3  # Strong bullish signal
                    entry_action = "buy"
                elif close < val:
                    entry_score += 3  # Strong bearish signal  
                    entry_action = "sell"
                elif close > poc:
                    entry_score += 1  # Weak bullish signal
                    entry_action = "buy"
                elif close < poc:
                    entry_score += 1  # Weak bearish signal
                    entry_action = "sell"
                
                # Trend confirmation (secondary)
                if entry_action == "buy" and close > self.sma50[data][0]:
                    entry_score += 2
                elif entry_action == "sell" and close < self.sma50[data][0]:
                    entry_score += 2
                
                # Volume confirmation (tertiary)
                if data.volume[0] > self.volume_ma[data][0]:
                    entry_score += 1
                
                # Rotation confirmation (bonus)
                if (entry_action == "buy" and rotation == "up") or (entry_action == "sell" and rotation == "down"):
                    entry_score += 1
                
                # Apply relaxation factor to lower threshold
                required_score = max(1, int(4 * self.relaxation_factor))
                
                # Execute trade if score meets threshold
                if entry_score >= required_score:
                    # Adjust position size based on signal strength
                    if entry_score >= 6:
                        entry_size = pos_size  # Full position
                    elif entry_score >= 4:
                        entry_size = int(pos_size * 0.7)  # 70% position
                    else:
                        entry_size = int(pos_size * 0.5)  # 50% position
                    
                    entry_signal = True
                
                # Execute entry signal (with holiday check)
                if entry_signal and entry_action:
                    # Check if order can be safely executed (no holiday on next day)
                    if not self._can_execute_order(data):
                        print(f"SKIPPED ENTRY: {entry_action.upper()} signal blocked due to upcoming holiday/missing data")
                        return
                    
                    if entry_action == "buy":
                        self.buy(data=data, size=entry_size)
                        print(f"ENTRY: Buy {entry_size} shares (score: {entry_score}, threshold: {required_score})")
                    elif entry_action == "sell":
                        self.sell(data=data, size=entry_size)
                        print(f"ENTRY: Sell {entry_size} shares (score: {entry_score}, threshold: {required_score})")
            
            elif position > 0:  # Long position - More aggressive exit logic
                exit_score = 0
                exit_reasons = []
                
                # Critical exits (immediate close)
                if close < (val - self.atr[data][0]):
                    exit_score += 10
                    exit_reasons.append("hard_stop_loss")
                elif close < val:
                    exit_score += 6  # Reduced from 8 to make exits easier
                    exit_reasons.append("value_area_breakdown")
                
                # Trend-based exits
                if close < self.sma50[data][0]:
                    exit_score += 4  # Increased from 3
                    exit_reasons.append("trend_reversal")
                
                # Volume-based exits
                if data.volume[0] > self.volume_ma[data][0] * 1.5 and close < poc:
                    exit_score += 3  # Increased from 2
                    exit_reasons.append("high_volume_selling")
                
                # Time-based exits (more aggressive)
                bars_held = self.bars_processed - self.last_trade_bar
                if bars_held > 50:  # Reduced from 100 to 50
                    exit_score += 8  # Increased from 5
                    exit_reasons.append("time_limit")
                elif bars_held > 25:  # Reduced from 50 to 25
                    exit_score += 4  # Increased from 2
                    exit_reasons.append("long_hold")
                elif bars_held > 15:  # New shorter time exit
                    exit_score += 2
                    exit_reasons.append("hold_time")
                
                # More aggressive profit-taking
                profit_pct = (close - data.close[-bars_held if bars_held > 0 else -1]) / data.close[-bars_held if bars_held > 0 else -1]
                if profit_pct > 0.02:  # Reduced from 3% to 2%
                    if excess == "up":
                        # Check if partial exit can be safely executed
                        if self._can_execute_order(data):
                            self.sell(data=data, size=int(position * 0.5))
                            print(f"PARTIAL EXIT: Taking 50% profits at {profit_pct:.2%} gain (excess move)")
                        else:
                            print(f"SKIPPED PARTIAL EXIT: Holiday/missing data detected")
                    elif profit_pct > 0.03:  # Reduced from 5% to 3%
                        exit_score += 4  # Increased from 3
                        exit_reasons.append("profit_target")
                
                # Apply relaxation factor to exit threshold (make exits easier)
                exit_threshold = max(3, int(6 * self.relaxation_factor))  # Reduced from 5,8 to 3,6
                
                if exit_score >= exit_threshold:
                    # Check if exit can be safely executed
                    if self._can_execute_order(data):
                        self.close(data=data)
                        print(f"LONG EXIT: Score {exit_score} >= {exit_threshold} (reasons: {', '.join(exit_reasons)})")
                    else:
                        print(f"SKIPPED LONG EXIT: Holiday/missing data detected (score: {exit_score}, threshold: {exit_threshold})")
            
            elif position < 0:  # Short position - More aggressive exit logic
                exit_score = 0
                exit_reasons = []
                
                # Critical exits (immediate close)
                if close > (vah + self.atr[data][0]):
                    exit_score += 10
                    exit_reasons.append("hard_stop_loss")
                elif close > vah:
                    exit_score += 6  # Reduced from 8 to make exits easier
                    exit_reasons.append("value_area_breakout")
                
                # Trend-based exits
                if close > self.sma50[data][0]:
                    exit_score += 4  # Increased from 3
                    exit_reasons.append("trend_reversal")
                
                # Volume-based exits
                if data.volume[0] > self.volume_ma[data][0] * 1.5 and close > poc:
                    exit_score += 3  # Increased from 2
                    exit_reasons.append("high_volume_buying")
                
                # Time-based exits (more aggressive)
                bars_held = self.bars_processed - self.last_trade_bar
                if bars_held > 50:  # Reduced from 100 to 50
                    exit_score += 8  # Increased from 5
                    exit_reasons.append("time_limit")
                elif bars_held > 25:  # Reduced from 50 to 25
                    exit_score += 4  # Increased from 2
                    exit_reasons.append("long_hold")
                elif bars_held > 15:  # New shorter time exit
                    exit_score += 2
                    exit_reasons.append("hold_time")
                
                # More aggressive profit-taking
                profit_pct = (data.close[-bars_held if bars_held > 0 else -1] - close) / data.close[-bars_held if bars_held > 0 else -1]
                if profit_pct > 0.02:  # Reduced from 3% to 2%
                    if excess == "down":
                        # Check if partial exit can be safely executed
                        if self._can_execute_order(data):
                            self.buy(data=data, size=int(abs(position) * 0.5))
                            print(f"PARTIAL EXIT: Taking 50% profits at {profit_pct:.2%} gain (excess move)")
                        else:
                            print(f"SKIPPED PARTIAL EXIT: Holiday/missing data detected")
                    elif profit_pct > 0.03:  # Reduced from 5% to 3%
                        exit_score += 4  # Increased from 3
                        exit_reasons.append("profit_target")
                
                # Apply relaxation factor to exit threshold (make exits easier)
                exit_threshold = max(3, int(6 * self.relaxation_factor))  # Reduced from 5,8 to 3,6
                
                if exit_score >= exit_threshold:
                    # Check if exit can be safely executed
                    if self._can_execute_order(data):
                        self.close(data=data)
                        print(f"SHORT EXIT: Score {exit_score} >= {exit_threshold} (reasons: {', '.join(exit_reasons)})")
                    else:
                        print(f"SKIPPED SHORT EXIT: Holiday/missing data detected (score: {exit_score}, threshold: {exit_threshold})")
        except Exception as e:
            print(f"Error applying auction market logic: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_valid_price_data(self, data, offset=0):
        """
        Validate that price data is valid for trading.
        
        Args:
            data: Backtrader data feed
            offset: Bar offset (0 for current, -1 for next, etc.)
            
        Returns:
            bool: True if data is valid for trading
        """
        try:
            # Get price data at specified offset
            close = data.close[offset]
            open_price = data.open[offset]
            high = data.high[offset] 
            low = data.low[offset]
            volume = data.volume[offset]
            
            # Check for zero, None, or NaN values
            if (close is None or open_price is None or high is None or 
                low is None or volume is None):
                return False
                
            # Check for zero prices (indicates missing data)
            if (close <= 0 or open_price <= 0 or high <= 0 or low <= 0):
                return False
                
            # Check for NaN values
            import math
            if any(math.isnan(val) if isinstance(val, (int, float)) else False 
                   for val in [close, open_price, high, low, volume]):
                return False
                
            # Check basic price logic (high >= low, close within range)
            if high < low or close < 0:
                return False
                
            # Check volume is reasonable
            if volume < 0:
                return False
                
            return True
            
        except Exception as e:
            # If we can't access the data (e.g., future bar doesn't exist), return False
            return False

    def _can_execute_order(self, data):
        """
        Check if an order can be safely executed by validating current and next bar's data.
        This prevents orders from being executed on holidays/missing data days.
        
        Args:
            data: Backtrader data feed
            
        Returns:
            bool: True if order can be safely executed
        """
        try:
            current_date = data.datetime.date(0)
            
            # First check if current bar has valid price data
            if not self._is_valid_price_data(data, offset=0):
                print(f"WARNING: Current trading day ({current_date}) has invalid price data. Skipping order to prevent holiday execution.")
                return False
            
            # Check if we have a next bar available
            if len(data) < 2:  # Need at least current and next bar
                return True  # Allow trading near the end of data
            
            # Check if next bar has valid price data
            if not self._is_valid_price_data(data, offset=-1):
                next_date = data.datetime.date(-1) if len(data) > 1 else None
                print(f"WARNING: Next trading day ({next_date}) has invalid price data. Skipping order on {current_date} to prevent holiday execution.")
                return False
                
            return True
            
        except Exception as e:
            # If in doubt, allow the order (fail open rather than fail closed)
            return True
    
    def _cancel_pending_orders(self):
        """Cancel all pending orders to prevent execution on invalid price data."""
        try:
            # Get all pending orders
            pending_orders = [order for order in self.broker.orders if order.status in [order.Submitted, order.Accepted]]
            
            for order in pending_orders:
                self.cancel(order)
                order_date = bt.num2date(order.created.dt).strftime('%Y-%m-%d') if hasattr(order, 'created') else "unknown"
                print(f"CANCELLED ORDER: {order.ordtype} order from {order_date} to prevent holiday execution")
                
        except Exception as e:
            print(f"Error cancelling pending orders: {e}")
    
    def notify_order(self, order):
        """Override to prevent execution of orders with invalid price data."""
        if order.status == order.Completed:
            # Check if the execution price is invalid (zero or negative)
            exec_price = order.executed.price
            if exec_price <= 0:
                exec_date = bt.num2date(order.executed.dt).strftime('%Y-%m-%d')
                print(f"INVALID ORDER EXECUTION: Order executed with price {exec_price} on {exec_date}")
                
                # Try to immediately reverse the invalid trade if possible
                try:
                    if order.executed.size > 0:  # Was a buy order
                        print(f"REVERSING INVALID BUY: Selling {order.executed.size} shares")
                        self.sell(size=order.executed.size)
                    else:  # Was a sell order
                        print(f"REVERSING INVALID SELL: Buying {abs(order.executed.size)} shares")
                        self.buy(size=abs(order.executed.size))
                except Exception as e:
                    print(f"Error reversing invalid order: {e}")
                
                return  # Don't process this invalid order further
        
        # Call parent implementation for valid orders
        super().notify_order(order)

    def _update_parameter_relaxation(self):
        """Enhanced parameter relaxation based on trading frequency and performance."""
        # Update bars without trade
        self.bars_without_trade = self.bars_processed - self.last_trade_bar
        
        # Progressive relaxation stages
        if self.bars_without_trade > 20:  # Start relaxing earlier
            if self.bars_without_trade <= 50:
                # Stage 1: Light relaxation (20-50 bars)
                self.relaxation_factor = 0.9
            elif self.bars_without_trade <= 100:
                # Stage 2: Moderate relaxation (50-100 bars)
                self.relaxation_factor = 0.7
            elif self.bars_without_trade <= 200:
                # Stage 3: Heavy relaxation (100-200 bars)
                self.relaxation_factor = 0.5
            else:
                # Stage 4: Maximum relaxation (200+ bars)
                self.relaxation_factor = self.max_relaxation
            
            # Log relaxation changes
            if self.bars_without_trade in [20, 50, 100, 200] or self.bars_without_trade % 100 == 0:
                print(f"Bar {self.bars_processed}: No trades for {self.bars_without_trade} bars. "
                      f"Relaxation factor: {self.relaxation_factor:.2f} (Stage {self._get_relaxation_stage()})")
        else:
            self.relaxation_factor = 1.0
    
    def _get_relaxation_stage(self):
        """Get current relaxation stage for logging."""
        if self.bars_without_trade <= 50:
            return 1
        elif self.bars_without_trade <= 100:
            return 2
        elif self.bars_without_trade <= 200:
            return 3
        else:
            return 4
    
    def _get_relaxed_threshold(self, original_threshold):
        """Apply relaxation factor to a threshold value."""
        return original_threshold * self.relaxation_factor
    
    def notify_order(self, order):
        """Log order execution information"""
        if order.status in [order.Completed]:
            if order.isbuy():
                action = "BUY"
            else:
                action = "SELL"
            
            # Update last trade bar
            self.last_trade_bar = self.bars_processed
            self.bars_without_trade = 0
            
            # Reset relaxation factor after successful trade
            if self.relaxation_factor < 1.0:
                print(f"Trade executed - resetting relaxation factor from {self.relaxation_factor:.2f} to 1.0")
                self.relaxation_factor = 1.0
            
            print(f"{self.data.datetime.date(0)} - {action} order executed: "
                  f"Price={order.executed.price:.2f}, Size={order.executed.size}, "
                  f"Value=${order.executed.value:.2f}, Commission=${order.executed.comm:.2f}") 