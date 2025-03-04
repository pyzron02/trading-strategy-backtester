#!/usr/bin/env python3
# auction_market_parameters.py - Define parameters for Auction Market Theory strategy

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