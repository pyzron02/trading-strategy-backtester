#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Performance Metrics - Provides better analysis for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class EnhancedPerformanceAnalyzer:
    """
    Enhanced performance analysis for trading strategies.
    """
    
    def __init__(self, equity_curve: pd.DataFrame, trade_log: pd.DataFrame):
        """
        Initialize with equity curve and trade log.
        
        Args:
            equity_curve: DataFrame with Date and Value columns
            trade_log: DataFrame with trade information
        """
        self.equity_curve = equity_curve.copy()
        self.trade_log = trade_log.copy()
        
        # Prepare data
        self.equity_curve['Date'] = pd.to_datetime(self.equity_curve['Date'])
        self.equity_curve = self.equity_curve.sort_values('Date').reset_index(drop=True)
        
        if len(self.trade_log) > 0:
            self.trade_log['date'] = pd.to_datetime(self.trade_log['date'])
    
    def diagnose_strategy_issues(self) -> Dict[str, Any]:
        """
        Diagnose common strategy implementation issues.
        
        Returns:
            Dict containing diagnosis results and recommendations
        """
        diagnosis = {
            'issues_found': [],
            'recommendations': [],
            'metrics': {},
            'trade_analysis': {},
            'risk_analysis': {}
        }
        
        # Analyze trade frequency
        total_trades = len(self.trade_log)
        trading_days = len(self.equity_curve)
        
        diagnosis['metrics']['total_trades'] = total_trades
        diagnosis['metrics']['trading_days'] = trading_days
        diagnosis['metrics']['trades_per_day'] = total_trades / trading_days if trading_days > 0 else 0
        
        # Issue 1: Too few trades (buy-and-hold behavior)
        if total_trades <= 2:
            diagnosis['issues_found'].append({
                'issue': 'Buy-and-Hold Behavior',
                'severity': 'Critical',
                'description': f'Only {total_trades} trades executed. Strategy behaves like buy-and-hold.',
                'impact': 'Equity curve tracks underlying asset, not strategy performance'
            })
            diagnosis['recommendations'].append(
                'Relax entry conditions or review strategy parameters to increase trading frequency'
            )
        elif total_trades < 10:
            diagnosis['issues_found'].append({
                'issue': 'Low Trading Frequency', 
                'severity': 'High',
                'description': f'Only {total_trades} trades. Insufficient for statistical significance.',
                'impact': 'Results may not be representative of strategy performance'
            })
        
        # Issue 2: Unmatched trades (opens without closes)
        if len(self.trade_log) > 0:
            open_trades = self.trade_log[self.trade_log['type'] == 'open']
            close_trades = self.trade_log[self.trade_log['type'] == 'close']
            
            diagnosis['trade_analysis']['open_trades'] = len(open_trades)
            diagnosis['trade_analysis']['close_trades'] = len(close_trades)
            diagnosis['trade_analysis']['unmatched_trades'] = abs(len(open_trades) - len(close_trades))
            
            if len(open_trades) != len(close_trades):
                diagnosis['issues_found'].append({
                    'issue': 'Unmatched Trades',
                    'severity': 'High', 
                    'description': f'{len(open_trades)} opens vs {len(close_trades)} closes',
                    'impact': 'Strategy may have open positions or exit logic issues'
                })
                diagnosis['recommendations'].append(
                    'Review exit conditions and add time-based stops to ensure positions close'
                )
        
        # Issue 3: Excessive volatility (spiky equity curve)
        if len(self.equity_curve) > 1:
            returns = self.equity_curve['Value'].pct_change().dropna()
            daily_vol = returns.std()
            max_daily_change = max(abs(returns.max()), abs(returns.min()))
            
            diagnosis['risk_analysis']['daily_volatility'] = daily_vol
            diagnosis['risk_analysis']['max_daily_change'] = max_daily_change
            diagnosis['risk_analysis']['annualized_volatility'] = daily_vol * np.sqrt(252)
            
            if max_daily_change > 0.05:  # > 5% daily change
                diagnosis['issues_found'].append({
                    'issue': 'Excessive Volatility',
                    'severity': 'Medium',
                    'description': f'Maximum daily change: {max_daily_change:.2%}',
                    'impact': 'Strategy may be taking excessive risk or have position sizing issues'
                })
                diagnosis['recommendations'].append(
                    'Review position sizing and implement proper risk management'
                )
        
        # Issue 4: Poor risk-adjusted returns
        if len(self.equity_curve) > 1:
            initial_value = self.equity_curve['Value'].iloc[0]
            final_value = self.equity_curve['Value'].iloc[-1]
            total_return = (final_value / initial_value) - 1
            
            # Calculate Sharpe ratio approximation
            returns = self.equity_curve['Value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_approx = returns.mean() / returns.std() * np.sqrt(252)
                diagnosis['metrics']['sharpe_ratio'] = sharpe_approx
                diagnosis['metrics']['total_return'] = total_return
                
                if sharpe_approx < 0.5:
                    diagnosis['issues_found'].append({
                        'issue': 'Poor Risk-Adjusted Returns',
                        'severity': 'Medium',
                        'description': f'Sharpe ratio: {sharpe_approx:.2f}',
                        'impact': 'Strategy may not provide adequate return for risk taken'
                    })
        
        return diagnosis
    
    def calculate_realistic_drawdowns(self) -> pd.DataFrame:
        """
        Calculate more realistic drawdowns that account for trading strategy behavior.
        
        Returns:
            DataFrame with enhanced drawdown calculations
        """
        dd_df = self.equity_curve.copy()
        
        # Standard rolling maximum drawdown
        dd_df['Peak'] = dd_df['Value'].expanding().max()
        dd_df['Drawdown'] = (dd_df['Value'] - dd_df['Peak']) / dd_df['Peak']
        
        # Trading-specific drawdowns
        if len(self.trade_log) > 0:
            # Mark periods with open positions
            dd_df['Has_Position'] = False
            
            open_trades = self.trade_log[self.trade_log['type'] == 'open']
            close_trades = self.trade_log[self.trade_log['type'] == 'close']
            
            for _, trade in open_trades.iterrows():
                start_date = trade['date']
                # Find corresponding close
                close_trade = close_trades[close_trades['date'] > start_date]
                end_date = close_trade['date'].min() if len(close_trade) > 0 else dd_df['Date'].max()
                
                mask = (dd_df['Date'] >= start_date) & (dd_df['Date'] <= end_date)
                dd_df.loc[mask, 'Has_Position'] = True
            
            # Calculate position-specific drawdowns
            position_periods = dd_df[dd_df['Has_Position']]
            if len(position_periods) > 0:
                # Drawdown during positions vs cash periods
                dd_df['Position_Drawdown'] = np.where(
                    dd_df['Has_Position'],
                    dd_df['Drawdown'],
                    0
                )
        
        return dd_df[['Date', 'Value', 'Peak', 'Drawdown']]
    
    def generate_enhanced_report(self) -> str:
        """
        Generate a comprehensive strategy analysis report.
        
        Returns:
            Formatted text report
        """
        diagnosis = self.diagnose_strategy_issues()
        
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED STRATEGY PERFORMANCE ANALYSIS")
        lines.append("=" * 80)
        
        # Summary metrics
        lines.append("\nKEY METRICS:")
        for metric, value in diagnosis['metrics'].items():
            if isinstance(value, float):
                if metric.endswith('ratio') or metric.endswith('return'):
                    lines.append(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    lines.append(f"  - {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
        
        # Issues found
        if diagnosis['issues_found']:
            lines.append(f"\nISSUES IDENTIFIED ({len(diagnosis['issues_found'])}):")
            for issue in diagnosis['issues_found']:
                lines.append(f"  🚨 {issue['severity'].upper()}: {issue['issue']}")
                lines.append(f"     Description: {issue['description']}")
                lines.append(f"     Impact: {issue['impact']}")
                lines.append("")
        
        # Recommendations
        if diagnosis['recommendations']:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(diagnosis['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
        
        # Trade analysis
        if diagnosis['trade_analysis']:
            lines.append(f"\nTRADE ANALYSIS:")
            for metric, value in diagnosis['trade_analysis'].items():
                lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
        
        # Risk analysis  
        if diagnosis['risk_analysis']:
            lines.append(f"\nRISK ANALYSIS:")
            for metric, value in diagnosis['risk_analysis'].items():
                if isinstance(value, float):
                    lines.append(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)