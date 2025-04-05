"""
Monte Carlo backtesting framework for trading strategies.
"""

from .trade_based_monte_carlo import TradeBasedMonteCarloTest
from .strategies import SimpleStock, MACrossover, AuctionMarket, MultiPosition
from .observers import PortfolioValue, TradeLog
from .utilities import save_to_json, CustomJSONEncoder

__all__ = [
    'TradeBasedMonteCarloTest',
    'SimpleStock', 
    'MACrossover', 
    'AuctionMarket', 
    'MultiPosition',
    'PortfolioValue',
    'TradeLog',
    'save_to_json',
    'CustomJSONEncoder'
]
