#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Health Monitor - Validates strategy performance and provides recommendations.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class StrategyHealthMonitor:
    """
    Monitors strategy health and provides warnings/recommendations.
    """
    
    def __init__(self, expected_metrics: Optional[Dict[str, float]] = None):
        """
        Initialize the health monitor.
        
        Args:
            expected_metrics: Dictionary of expected metric thresholds
        """
        self.expected_metrics = expected_metrics or self._get_default_metrics()
        
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default expected metrics for a healthy strategy."""
        return {
            'min_trades_per_year': 12,  # At least 1 trade per month
            'min_trade_frequency': 0.05,  # At least 5% of days should have trades
            'max_acceptable_drawdown': 0.25,  # 25% maximum drawdown
            'min_sharpe_ratio': 0.5,  # Minimum acceptable Sharpe ratio
            'min_win_rate': 0.35,  # At least 35% win rate
            'min_profit_factor': 1.0,  # Profits should exceed losses
            'max_consecutive_losses': 10,  # Maximum consecutive losing trades
            'min_unique_trades': 10  # Minimum number of trades for meaningful analysis
        }
    
    def check_strategy_health(self, 
                            strategy_name: str,
                            backtest_results: Dict[str, Any],
                            trades_df: Optional[Any] = None) -> Dict[str, Any]:
        """
        Check strategy health and provide comprehensive report.
        
        Args:
            strategy_name: Name of the strategy
            backtest_results: Dictionary containing backtest results
            trades_df: Optional DataFrame containing individual trades
            
        Returns:
            Dict containing health status, warnings, and recommendations
        """
        health_report = {
            'strategy_name': strategy_name,
            'status': 'healthy',
            'score': 100,  # Health score out of 100
            'warnings': [],
            'critical_issues': [],
            'recommendations': [],
            'metrics_summary': {}
        }
        
        # Extract key metrics
        total_trades = backtest_results.get('total_trades', 0)
        start_date = backtest_results.get('start_date')
        end_date = backtest_results.get('end_date')
        max_drawdown = abs(backtest_results.get('max_drawdown', 0))
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        win_rate = backtest_results.get('win_rate', 0)
        profit_factor = backtest_results.get('profit_factor', 0)
        
        # Calculate trading period in years
        if start_date and end_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            trading_days = (end_date - start_date).days
            trading_years = trading_days / 365.25
        else:
            trading_years = 1  # Default to 1 year if dates not available
            
        # Store metrics summary
        health_report['metrics_summary'] = {
            'total_trades': total_trades,
            'trading_period_years': round(trading_years, 2),
            'trades_per_year': round(total_trades / trading_years, 2) if trading_years > 0 else 0,
            'max_drawdown': round(max_drawdown, 4),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 2)
        }
        
        # Check 1: Minimum trades
        if total_trades < self.expected_metrics['min_unique_trades']:
            health_report['critical_issues'].append(
                f"Insufficient trades: {total_trades} trades. Need at least "
                f"{self.expected_metrics['min_unique_trades']} for meaningful analysis."
            )
            health_report['recommendations'].append(
                "Consider relaxing entry conditions or reviewing strategy parameters."
            )
            health_report['score'] -= 40
            health_report['status'] = 'critical'
            
        # Check 2: Trade frequency
        trades_per_year = total_trades / trading_years if trading_years > 0 else 0
        if trades_per_year < self.expected_metrics['min_trades_per_year']:
            severity = 'critical' if trades_per_year < 6 else 'warning'
            message = f"Low trading frequency: {trades_per_year:.1f} trades/year"
            
            if severity == 'critical':
                health_report['critical_issues'].append(message)
                health_report['score'] -= 30
            else:
                health_report['warnings'].append(message)
                health_report['score'] -= 15
                
            health_report['recommendations'].append(
                "Review strategy conditions - they may be too restrictive."
            )
            
            if health_report['status'] == 'healthy':
                health_report['status'] = severity
                
        # Check 3: Drawdown
        if max_drawdown > self.expected_metrics['max_acceptable_drawdown']:
            health_report['warnings'].append(
                f"High drawdown: {max_drawdown:.1%} exceeds acceptable limit of "
                f"{self.expected_metrics['max_acceptable_drawdown']:.0%}"
            )
            health_report['recommendations'].append(
                "Consider implementing tighter risk management or position sizing."
            )
            health_report['score'] -= 20
            
            if health_report['status'] == 'healthy':
                health_report['status'] = 'warning'
                
        # Check 4: Sharpe ratio
        if sharpe_ratio < self.expected_metrics['min_sharpe_ratio']:
            health_report['warnings'].append(
                f"Low Sharpe ratio: {sharpe_ratio:.2f} below minimum of "
                f"{self.expected_metrics['min_sharpe_ratio']}"
            )
            health_report['recommendations'].append(
                "Review risk-adjusted returns. Consider optimizing entry/exit timing."
            )
            health_report['score'] -= 15
            
        # Check 5: Win rate
        if win_rate < self.expected_metrics['min_win_rate']:
            health_report['warnings'].append(
                f"Low win rate: {win_rate:.1%} below minimum of "
                f"{self.expected_metrics['min_win_rate']:.0%}"
            )
            health_report['recommendations'].append(
                "Analyze losing trades to identify patterns. Consider refining entry signals."
            )
            health_report['score'] -= 10
            
        # Check 6: Profit factor
        if profit_factor < self.expected_metrics['min_profit_factor']:
            health_report['critical_issues'].append(
                f"Unprofitable strategy: profit factor {profit_factor:.2f} < 1.0"
            )
            health_report['recommendations'].append(
                "Strategy is losing money. Major revision needed."
            )
            health_report['score'] -= 30
            health_report['status'] = 'critical'
            
        # Ensure score doesn't go below 0
        health_report['score'] = max(0, health_report['score'])
        
        # Add general recommendations based on health score
        if health_report['score'] < 50:
            health_report['recommendations'].append(
                "Consider running parameter optimization before further analysis."
            )
            
        # Special case: very few trades but otherwise healthy
        if total_trades < 30 and health_report['status'] != 'critical':
            health_report['recommendations'].append(
                "Limited trade sample size. Results may not be statistically significant. "
                "Consider extending backtest period or testing on multiple instruments."
            )
            
        return health_report
    
    def generate_health_report_text(self, health_report: Dict[str, Any]) -> str:
        """
        Generate a formatted text report from health check results.
        
        Args:
            health_report: Health report dictionary from check_strategy_health
            
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"STRATEGY HEALTH REPORT: {health_report['strategy_name']}")
        lines.append("=" * 60)
        lines.append(f"Overall Status: {health_report['status'].upper()}")
        lines.append(f"Health Score: {health_report['score']}/100")
        lines.append("")
        
        # Metrics summary
        lines.append("Key Metrics:")
        for metric, value in health_report['metrics_summary'].items():
            lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Critical issues
        if health_report['critical_issues']:
            lines.append("CRITICAL ISSUES:")
            for issue in health_report['critical_issues']:
                lines.append(f"  ⚠️  {issue}")
            lines.append("")
            
        # Warnings
        if health_report['warnings']:
            lines.append("WARNINGS:")
            for warning in health_report['warnings']:
                lines.append(f"  ⚡ {warning}")
            lines.append("")
            
        # Recommendations
        if health_report['recommendations']:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(health_report['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def should_run_monte_carlo(self, health_report: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if Monte Carlo analysis should be run based on health report.
        
        Args:
            health_report: Health report from check_strategy_health
            
        Returns:
            Tuple of (should_run, reason)
        """
        # Don't run if critical issues exist
        if health_report['status'] == 'critical':
            return False, "Strategy has critical issues that should be addressed first"
            
        # Don't run if too few trades
        total_trades = health_report['metrics_summary'].get('total_trades', 0)
        if total_trades < 30:
            return False, f"Insufficient trades ({total_trades}). Need at least 30 for meaningful Monte Carlo analysis"
            
        # Warn but allow if status is warning
        if health_report['status'] == 'warning':
            return True, "Proceeding with warnings - results should be interpreted cautiously"
            
        return True, "Strategy is healthy for Monte Carlo analysis"