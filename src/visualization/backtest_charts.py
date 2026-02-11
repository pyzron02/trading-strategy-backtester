#!/usr/bin/env python3
"""
Backtest visualization module.

Generates interactive Plotly and static matplotlib visualizations
for backtest results including equity curves, drawdowns, candlestick
charts with trade markers, and performance dashboards.
"""

import os
import pandas as pd
import numpy as np

# Plotly imports for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Basic plots will use matplotlib. "
          "Install with 'pip install plotly'")

# Check if kaleido is available for saving static images
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def plot_connected_trades(fig, entries_df, exits_df, marker_color,
                          line_color, marker_symbol_entry,
                          marker_symbol_exit, name_prefix,
                          highlight_trades, row=1, col=1,
                          ticker_prefix=None, border_color='white',
                          use_kaleido_sizes=False):
    """
    Plot connected entry/exit trade markers on a Plotly figure.

    Draws lines connecting each entry-exit pair with configurable
    marker symbols, colors, and sizes. This is a unified function
    that replaces the three inline variants (plot_connected_trades_ticker,
    plot_connected_trades_main, plot_connected_trades).

    Args:
        fig: Plotly Figure or subplot figure to add traces to.
        entries_df: DataFrame of entry trades with 'date' and 'price' columns.
        exits_df: DataFrame of exit trades with 'date', 'price', and
            optionally 'pnl' columns.
        marker_color: Color string for the markers.
        line_color: Color string for the connecting lines.
        marker_symbol_entry: Plotly marker symbol for entry points.
        marker_symbol_exit: Plotly marker symbol for exit points.
        name_prefix: Prefix for legend labels (e.g. 'Long Trade').
        highlight_trades: If False, skip plotting entirely.
        row: Subplot row to add traces to.
        col: Subplot column to add traces to.
        ticker_prefix: Optional ticker name to prepend to legend labels.
        border_color: Color for the marker outline border.
        use_kaleido_sizes: If True, use slightly larger markers and
            thicker lines suited for static PNG export.

    Returns:
        None
    """
    if not highlight_trades:
        return

    if entries_df.empty or exits_df.empty:
        return

    # Determine marker/line sizes
    if use_kaleido_sizes:
        marker_size_entry = 14
        marker_size_exit = 14
        line_width = 2.5
        border_width = 2
    else:
        marker_size_entry = 18
        marker_size_exit = 14
        line_width = 2.5
        border_width = 2

    # Sort entries and exits by date
    entries_df = entries_df.sort_values('date')
    exits_df = exits_df.sort_values('date')

    # Match entry-exit pairs by order, taking the minimum count
    min_trades = min(len(entries_df), len(exits_df))
    if min_trades == 0:
        return

    entries_df = entries_df.iloc[:min_trades].reset_index(drop=True)
    exits_df = exits_df.iloc[:min_trades].reset_index(drop=True)

    for i in range(min_trades):
        entry_date = entries_df['date'].iloc[i]
        entry_price = entries_df['price'].iloc[i]
        exit_date = exits_df['date'].iloc[i]
        exit_price = exits_df['price'].iloc[i]
        pnl = exits_df['pnl'].iloc[i] if 'pnl' in exits_df.columns else 0

        label = (f'{ticker_prefix} {name_prefix} {i+1}'
                 if ticker_prefix else f'{name_prefix} {i+1}')

        fig.add_trace(
            go.Scatter(
                x=[entry_date, exit_date],
                y=[entry_price, exit_price],
                mode='lines+markers',
                name=label,
                line=dict(color=line_color, width=line_width, dash='dot'),
                marker=dict(
                    color=marker_color,
                    size=[marker_size_entry, marker_size_exit],
                    symbol=[marker_symbol_entry, marker_symbol_exit],
                    line=dict(width=border_width, color=border_color)
                ),
                hoverinfo='text',
                hovertext=[
                    f'{name_prefix} Entry<br>Date: {entry_date}<br>'
                    f'Price: ${entry_price:.2f}',
                    f'{name_prefix} Exit<br>Date: {exit_date}<br>'
                    f'Price: ${exit_price:.2f}<br>PnL: ${pnl:.2f}'
                ],
                showlegend=(i == 0)
            ),
            row=row, col=col
        )


def _classify_trades(open_trades, closed_trades):
    """
    Classify open/closed trades into long and short entry/exit groups.

    Args:
        open_trades: DataFrame of opening trades.
        closed_trades: DataFrame of closing trades.

    Returns:
        Tuple of (buy_entries, sell_exits, sell_entries, buy_exits)
            DataFrames.
    """
    if 'action' in open_trades.columns:
        buy_entries = open_trades[
            open_trades['action'].str.contains('buy', case=False, na=False)]
        sell_exits = closed_trades[
            closed_trades['action'].str.contains('sell', case=False, na=False)]
        sell_entries = open_trades[
            open_trades['action'].str.contains('sell', case=False, na=False)]
        buy_exits = closed_trades[
            closed_trades['action'].str.contains('buy', case=False, na=False)]
    elif 'signal' in open_trades.columns:
        buy_entries = open_trades[
            open_trades['signal'].str.contains('buy', case=False, na=False)]
        sell_exits = closed_trades[
            closed_trades['signal'].str.contains('sell', case=False, na=False)]
        sell_entries = open_trades[
            open_trades['signal'].str.contains('sell', case=False, na=False)]
        buy_exits = closed_trades[
            closed_trades['signal'].str.contains('buy', case=False, na=False)]
    else:
        # Default: open trades are buys, close trades are sells
        buy_entries = open_trades
        sell_exits = closed_trades
        sell_entries = pd.DataFrame()
        buy_exits = pd.DataFrame()

    return buy_entries, sell_exits, sell_entries, buy_exits


def _add_trade_markers_to_ticker_fig(ticker_fig, trade_log, ticker,
                                     highlight_trades):
    """
    Add connected trade markers to an individual ticker figure.

    Args:
        ticker_fig: The Plotly figure for a single ticker.
        trade_log: DataFrame of all trades.
        ticker: The ticker symbol to filter trades for.
        highlight_trades: Whether trade highlighting is enabled.

    Returns:
        None
    """
    if trade_log is None or trade_log.empty:
        return
    if 'ticker' not in trade_log.columns:
        return

    ticker_trades_df = trade_log[trade_log['ticker'] == ticker]
    if ticker_trades_df.empty:
        return

    open_trades = (ticker_trades_df[ticker_trades_df['type'] == 'open']
                   if 'type' in ticker_trades_df.columns
                   else pd.DataFrame())
    closed_trades = (ticker_trades_df[ticker_trades_df['type'] == 'close']
                     if 'type' in ticker_trades_df.columns
                     else pd.DataFrame())

    buy_entries, sell_exits, sell_entries, buy_exits = _classify_trades(
        open_trades, closed_trades)

    plot_connected_trades(
        ticker_fig, buy_entries, sell_exits,
        '#00CC00', 'rgba(0,204,0,0.8)',
        'triangle-up', 'circle', 'Long Trade',
        highlight_trades, row=1, col=1, border_color='white')

    plot_connected_trades(
        ticker_fig, sell_entries, buy_exits,
        '#FF3333', 'rgba(255,51,51,0.8)',
        'triangle-down', 'circle', 'Short Trade',
        highlight_trades, row=1, col=1, border_color='white')


def _add_trade_markers_to_main_fig(fig, trade_log, ticker_data,
                                   highlight_trades):
    """
    Add connected trade markers to the main combined figure.

    Args:
        fig: The main combined Plotly figure.
        trade_log: DataFrame of all trades.
        ticker_data: Dict of ticker data keyed by ticker symbol.
        highlight_trades: Whether trade highlighting is enabled.

    Returns:
        None
    """
    if trade_log is None or trade_log.empty:
        return

    for ticker_idx, ticker in enumerate(ticker_data.keys(), 1):
        ticker_trades_df = (
            trade_log[trade_log['ticker'] == ticker]
            if 'ticker' in trade_log.columns
            else pd.DataFrame())

        if ticker_trades_df.empty:
            continue

        open_trades = (ticker_trades_df[ticker_trades_df['type'] == 'open']
                       if 'type' in ticker_trades_df.columns
                       else pd.DataFrame())
        closed_trades = (
            ticker_trades_df[ticker_trades_df['type'] == 'close']
            if 'type' in ticker_trades_df.columns
            else pd.DataFrame())

        buy_entries, sell_exits, sell_entries, buy_exits = _classify_trades(
            open_trades, closed_trades)

        plot_connected_trades(
            fig, buy_entries, sell_exits,
            '#00CC00', 'rgba(0,204,0,0.8)',
            'triangle-up', 'circle', 'Long Trade',
            highlight_trades, row=ticker_idx, col=1,
            ticker_prefix=ticker, border_color='white')

        plot_connected_trades(
            fig, sell_entries, buy_exits,
            '#FF3333', 'rgba(255,51,51,0.8)',
            'triangle-down', 'circle', 'Short Trade',
            highlight_trades, row=ticker_idx, col=1,
            ticker_prefix=ticker, border_color='white')


def _create_individual_ticker_plots_first_pass(
        ticker_data, strategy_name, trade_log, highlight_trades,
        ticker_plots_dir):
    """
    Create individual ticker plots during the first pass through
    ticker data (inside the main figure loop).

    Args:
        ticker_data: Dict of ticker OHLCV data keyed by ticker symbol.
        strategy_name: Name of the strategy.
        trade_log: DataFrame of all trades (or None).
        highlight_trades: Whether trade highlighting is enabled.
        ticker_plots_dir: Directory to save individual ticker plots.

    Returns:
        None
    """
    for ticker, data in ticker_data.items():
        ticker_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
            subplot_titles=[f'{ticker} Price', 'Volume']
        )

        ticker_fig.add_trace(
            go.Candlestick(
                x=data['dates'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=ticker,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        if (data['volume'] is not None
                and any(v > 0 for v in data['volume'])):
            ticker_fig.add_trace(
                go.Bar(
                    x=data['dates'],
                    y=data['volume'],
                    name=f'{ticker} Volume',
                    marker_color='rgba(0,0,0,0.2)',
                    showlegend=False
                ),
                row=2, col=1
            )

        _add_trade_markers_to_ticker_fig(
            ticker_fig, trade_log, ticker, highlight_trades)

        ticker_fig.update_layout(
            title=f'{strategy_name}: {ticker} Analysis',
            height=700,
            width=1200,
            template='plotly_white',
            hovermode='x unified'
        )

        ticker_html_path = os.path.join(
            ticker_plots_dir, f'{ticker}_plot.html')
        ticker_fig.write_html(
            ticker_html_path, config={'responsive': True})
        print(f"Saved individual plot for {ticker} to {ticker_html_path}")

        if KALEIDO_AVAILABLE:
            ticker_png_path = os.path.join(
                ticker_plots_dir, f'{ticker}_plot.png')
            ticker_fig.write_image(
                ticker_png_path, width=1200, height=700, scale=2)
            print(f"Saved static image for {ticker} to {ticker_png_path}")


def _create_individual_ticker_plots_second_pass(
        ticker_data, strategy_name, trade_log, highlight_trades,
        ticker_plots_dir):
    """
    Create individual ticker plots during the second pass (after the
    main figure is saved). Includes enhanced PNG markers.

    Args:
        ticker_data: Dict of ticker OHLCV data keyed by ticker symbol.
        strategy_name: Name of the strategy.
        trade_log: DataFrame of all trades (or None).
        highlight_trades: Whether trade highlighting is enabled.
        ticker_plots_dir: Directory to save individual ticker plots.

    Returns:
        None
    """
    if trade_log is None or trade_log.empty:
        return

    for ticker, data in ticker_data.items():
        ticker_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
            subplot_titles=[f'{ticker} Price', 'Volume']
        )

        ticker_fig.add_trace(
            go.Candlestick(
                x=data['dates'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=ticker,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        if (data['volume'] is not None
                and any(v > 0 for v in data['volume'])):
            ticker_fig.add_trace(
                go.Bar(
                    x=data['dates'],
                    y=data['volume'],
                    name=f'{ticker} Volume',
                    marker_color='rgba(0,0,0,0.2)',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Filter trades for this ticker
        ticker_trades_df = (
            trade_log[trade_log['ticker'] == ticker]
            if 'ticker' in trade_log.columns
            else pd.DataFrame())

        if not ticker_trades_df.empty:
            open_trades = (
                ticker_trades_df[ticker_trades_df['type'] == 'open']
                if 'type' in ticker_trades_df.columns
                else pd.DataFrame())
            closed_trades = (
                ticker_trades_df[ticker_trades_df['type'] == 'close']
                if 'type' in ticker_trades_df.columns
                else pd.DataFrame())

            buy_entries, sell_exits, sell_entries, buy_exits = (
                _classify_trades(open_trades, closed_trades))

            # Use the unified plot_connected_trades with kaleido-aware
            # sizes via use_kaleido_sizes flag
            plot_connected_trades(
                ticker_fig, buy_entries, sell_exits,
                '#00CC00', 'rgba(0,204,0,0.8)',
                'triangle-up', 'circle', 'Long Trade',
                highlight_trades, row=1, col=1,
                border_color='black',
                use_kaleido_sizes=KALEIDO_AVAILABLE)

            plot_connected_trades(
                ticker_fig, sell_entries, buy_exits,
                '#FF3333', 'rgba(255,51,51,0.8)',
                'triangle-down', 'circle', 'Short Trade',
                highlight_trades, row=1, col=1,
                border_color='black',
                use_kaleido_sizes=KALEIDO_AVAILABLE)

            # Additional marker/line sizing variables for the
            # direct trade-pair loop below
            marker_size_entry = 12
            marker_size_exit = 12
            line_width = 2.0
            border_width = 1.5
            if KALEIDO_AVAILABLE:
                marker_size_entry = 14
                marker_size_exit = 14
                line_width = 2.5
                border_width = 2

            # Direct trade pair plotting (open/close matching)
            if len(open_trades) > 0 and len(closed_trades) > 0:
                min_trades = min(len(open_trades), len(closed_trades))
                for i in range(min_trades):
                    entry = open_trades.iloc[i]
                    exit_t = closed_trades.iloc[i]

                    is_long = True
                    if is_long:
                        trade_color = '#00CC00'
                        tc_line_color = 'rgba(0,204,0,0.5)'
                        entry_symbol = 'triangle-up'
                    else:
                        trade_color = '#FF3333'
                        tc_line_color = 'rgba(255,51,51,0.5)'
                        entry_symbol = 'triangle-down'

                    ticker_fig.add_trace(
                        go.Scatter(
                            x=[entry['date'], exit_t['date']],
                            y=[entry['price'], exit_t['price']],
                            mode='lines+markers',
                            name=(f"{'Long' if is_long else 'Short'}"
                                  f" Trade {i+1}"),
                            line=dict(color=tc_line_color,
                                      width=line_width, dash='solid'),
                            marker=dict(
                                color=trade_color,
                                size=[marker_size_entry, marker_size_exit],
                                symbol=[entry_symbol, 'circle'],
                                line=dict(width=border_width,
                                          color='black')
                            ),
                            hoverinfo='text',
                            hovertext=[
                                (f"{'Long' if is_long else 'Short'} Entry"
                                 f"<br>Date: {entry['date']}"
                                 f"<br>Price: ${entry['price']:.2f}"),
                                (f"{'Long' if is_long else 'Short'} Exit"
                                 f"<br>Date: {exit_t['date']}"
                                 f"<br>Price: ${exit_t['price']:.2f}"
                                 f"<br>PnL: "
                                 f"${exit_t.get('pnl', 0):.2f}")
                            ],
                            showlegend=(i == 0),
                            legendgroup=(
                                'Long' if is_long else 'Short')
                        ),
                        row=1, col=1
                    )

        ticker_fig.update_layout(
            title=f'{strategy_name}: {ticker} Analysis',
            height=700,
            width=1200,
            template='plotly_white',
            hovermode='x unified'
        )

        ticker_html_path = os.path.join(
            ticker_plots_dir, f'{ticker}_plot.html')
        ticker_fig.write_html(
            ticker_html_path, config={'responsive': True})
        print(f"Saved individual plot for {ticker} to {ticker_html_path}")

        if KALEIDO_AVAILABLE:
            png_fig = ticker_fig

            # Add extra standalone markers for PNG output
            if trade_log is not None and not trade_log.empty:
                ticker_trades = (
                    trade_log[trade_log['ticker'] == ticker]
                    if 'ticker' in trade_log.columns
                    else pd.DataFrame())
                if not ticker_trades.empty:
                    entries = ticker_trades[
                        ticker_trades['type'] == 'open']
                    for _, row_data in entries.iterrows():
                        png_fig.add_trace(
                            go.Scatter(
                                x=[row_data['date']],
                                y=[row_data['price']],
                                mode='markers',
                                name='Entry',
                                marker=dict(
                                    color='#00CC00',
                                    size=14,
                                    symbol=(
                                        'triangle-down'
                                        if 'sell' in str(
                                            row_data['action']).lower()
                                        else 'triangle-up'),
                                    line=dict(width=2, color='black')
                                ),
                                showlegend=False
                            ),
                            row=1, col=1
                        )

                    exits = ticker_trades[
                        ticker_trades['type'] == 'close']
                    for _, row_data in exits.iterrows():
                        png_fig.add_trace(
                            go.Scatter(
                                x=[row_data['date']],
                                y=[row_data['price']],
                                mode='markers',
                                name='Exit',
                                marker=dict(
                                    color='#FF3333',
                                    size=14,
                                    symbol='circle',
                                    line=dict(width=2, color='black')
                                ),
                                showlegend=False
                            ),
                            row=1, col=1
                        )

            # Enhance existing markers and lines for PNG
            for trace in png_fig.data:
                if (isinstance(trace, go.Scatter)
                        and trace.mode):
                    if 'lines' in trace.mode and trace.line:
                        trace.line.width = 2.0
                        trace.line.dash = 'solid'
                    if ('markers' in trace.mode
                            and trace.marker):
                        if (hasattr(trace.marker, 'line')
                                and trace.marker.line):
                            trace.marker.line.width = 1.5
                            trace.marker.line.color = 'black'

            ticker_png_path = os.path.join(
                ticker_plots_dir, f'{ticker}_plot.png')
            png_fig.write_image(
                ticker_png_path, width=1200, height=700, scale=3)
            print(f"Saved greatly enhanced static image for "
                  f"{ticker} to {ticker_png_path}")


def generate_backtest_visualizations(
    output_dir,
    strategy_name,
    tickers,
    equity_curve,
    drawdowns,
    trade_log,
    ticker_data,
    metrics,
    cerebro,
    strategy,
    highlight_trades,
    enhanced_plots
):
    """
    Generate all backtest visualizations.

    Creates interactive Plotly charts (candlestick with trade markers,
    equity curve, drawdowns) and a performance dashboard. Falls back
    to backtrader native plotting when Plotly is unavailable.

    This function should only be called when plotting is requested
    (i.e. the caller checks ``plot == True`` before calling).

    Args:
        output_dir: Directory to save visualization files.
        strategy_name: Name of the strategy being backtested.
        tickers: List of ticker symbols.
        equity_curve: DataFrame with equity curve data, or None.
        drawdowns: Series of drawdown values, or None.
        trade_log: DataFrame of trade log entries, or None.
        ticker_data: Dict mapping ticker symbols to dicts with keys
            'dates', 'open', 'high', 'low', 'close', 'volume'.
        metrics: Dict of performance metrics.
        cerebro: The backtrader Cerebro instance.
        strategy: The strategy instance from the backtest run.
        highlight_trades: Whether to highlight individual trades.
        enhanced_plots: Whether to generate enhanced plots.

    Returns:
        None
    """
    try:
        if PLOTLY_AVAILABLE:
            _generate_plotly_visualizations(
                output_dir, strategy_name, tickers,
                equity_curve, drawdowns, trade_log,
                ticker_data, metrics, cerebro, strategy,
                highlight_trades, enhanced_plots)
        else:
            # If Plotly is not available, use backtrader's plotting
            print("Plotly is not available, using backtrader's "
                  "native plotting")
            fig = cerebro.plot(
                style='candle', barup='green', bardown='red',
                volume=False, grid=True)
            plot_file = os.path.join(
                output_dir,
                f"{strategy_name}_backtest_plot.png")
            if isinstance(fig, list) and len(fig) > 0:
                fig[0][0].savefig(plot_file)
                print(f"Saved plot to {plot_file}")
            elif hasattr(fig, 'savefig'):
                fig.savefig(plot_file)
                print(f"Saved plot to {plot_file}")
    except Exception as e:
        import traceback
        print(f"Error generating plot: {e}")
        traceback.print_exc()


def _generate_plotly_visualizations(
    output_dir, strategy_name, tickers,
    equity_curve, drawdowns, trade_log,
    ticker_data, metrics, cerebro, strategy,
    highlight_trades, enhanced_plots
):
    """
    Internal: generate all Plotly-based visualizations.

    Args:
        (Same as generate_backtest_visualizations)

    Returns:
        None
    """
    # Get the equity curve data
    equity_curve_data = None
    if (equity_curve is not None
            and not equity_curve.empty):
        equity_curve_data = {
            'dates': (equity_curve['Date']
                      if 'Date' in equity_curve.columns
                      else list(range(len(equity_curve)))),
            'values': (equity_curve['Value']
                       if 'Value' in equity_curve.columns
                       else equity_curve.iloc[:, 0])
        }
        # Calculate drawdowns from equity curve
        running_max = pd.Series(
            equity_curve_data['values']).cummax()
        ec_drawdowns = (
            equity_curve_data['values'] / running_max) - 1

        # Create subplot layout
        num_tickers = len(ticker_data)
        fig_height = 250 * (num_tickers + 1)
        row_heights = ([0.7 / num_tickers] * num_tickers
                       + [0.3])
        subplot_titles = (list(ticker_data.keys())
                          + ['Equity Curve'])

        # Create plots directory for individual ticker plots
        ticker_plots_dir = os.path.join(
            output_dir, 'ticker_plots')
        os.makedirs(ticker_plots_dir, exist_ok=True)

        # Create main combined figure
        fig = make_subplots(
            rows=num_tickers + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )

        # Add OHLC/candlestick charts for each ticker
        row = 1
        for ticker, data in ticker_data.items():
            fig.add_trace(
                go.Candlestick(
                    x=data['dates'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=ticker,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=row, col=1
            )

            if (data['volume'] is not None
                    and any(v > 0 for v in data['volume'])):
                fig.add_trace(
                    go.Bar(
                        x=data['dates'],
                        y=data['volume'],
                        name=f'{ticker} Volume',
                        marker_color='rgba(0,0,0,0.2)',
                        showlegend=False,
                        yaxis=f'y{row*2}'
                    ),
                    row=row, col=1
                )

                fig.update_layout({
                    f'yaxis{row*2-1}': {
                        'domain': [0.3, 1.0],
                        'title': 'Price'},
                    f'yaxis{row*2}': {
                        'domain': [0, 0.2],
                        'title': 'Volume',
                        'showgrid': False,
                        'anchor': f'x{row}'}
                })

            row += 1

        # Create individual ticker plots (first pass)
        _create_individual_ticker_plots_first_pass(
            ticker_data, strategy_name, trade_log,
            highlight_trades, ticker_plots_dir)

        # Add equity curve at the bottom
        fig.add_trace(
            go.Scatter(
                x=equity_curve_data['dates'],
                y=equity_curve_data['values'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2),
                hovertemplate=(
                    'Date: %{x}<br>Value: $%{y:.2f}'
                    '<extra></extra>')
            ),
            row=num_tickers + 1, col=1
        )

        # Add drawdown line
        fig.add_trace(
            go.Scatter(
                x=equity_curve_data['dates'],
                y=ec_drawdowns,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1.5),
                yaxis=f'y{(num_tickers+1)*2}',
                hovertemplate=(
                    'Date: %{x}<br>Drawdown: %{y:.2%}'
                    '<extra></extra>')
            ),
            row=num_tickers + 1, col=1
        )

        # Set up secondary y-axis for drawdowns
        fig.update_layout({
            f'yaxis{(num_tickers+1)*2-1}': {
                'title': 'Equity Value',
                'tickprefix': '$'},
            f'yaxis{(num_tickers+1)*2}': {
                'title': 'Drawdown',
                'tickformat': '%',
                'overlaying': f'y{(num_tickers+1)*2-1}',
                'side': 'right',
                'range': [min(ec_drawdowns.min() * 1.1, -0.05),
                          0.05]}
        })

        # Add trade markers to main figure
        _add_trade_markers_to_main_fig(
            fig, trade_log, ticker_data, highlight_trades)

        # Update layout with formatting
        fig.update_layout(
            title=f'{strategy_name}: Backtest Results',
            xaxis_title='Date',
            yaxis=dict(
                title='Equity Value',
                tickprefix='$',
                side='left',
                showgrid=True
            ),
            yaxis2=dict(
                title='Drawdown',
                tickformat='%',
                side='right',
                overlaying='y',
                range=[min(ec_drawdowns.min() * 1.1, -0.05),
                       0.05],
                showgrid=False,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.2)',
                zerolinewidth=1
            ),
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            width=1200,
            height=700,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white'
        )

        # Create dashboard with both charts and metrics table
        dashboard = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            specs=[
                [{'type': 'xy'}],
                [{'type': 'table'}]
            ],
            subplot_titles=(
                f'{strategy_name}: Equity Curve & Drawdowns',
                'Performance Metrics'
            ),
            vertical_spacing=0.12
        )

        for trace in fig.data:
            dashboard.add_trace(trace, row=1, col=1)

        dashboard.update_layout(
            yaxis=fig.layout.yaxis,
            yaxis2=fig.layout.yaxis2,
            legend=fig.layout.legend,
            width=1200,
            height=900,
            template='plotly_white',
            hovermode='x unified'
        )

        # Format metrics for the table (first instance)
        metrics_table = go.Table(
            header=dict(
                values=['Metric', 'Value',
                        'Trade Metrics', 'Value'],
                align=['left', 'right', 'left', 'right'],
                fill_color='royalblue',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    ['Initial Capital', 'Final Value',
                     'Total Return', 'Max Drawdown',
                     'Sharpe Ratio', 'Annual Return'],
                    [
                        f"${metrics.get('initial_value', 0):,.2f}",
                        f"${metrics.get('final_value', 0):,.2f}",
                        f"{metrics.get('total_return', 0):.2%}",
                        f"${metrics.get('max_drawdown_money', 0):,.2f}",
                        f"{metrics.get('sharpe_ratio', 0):.2f}",
                        f"{metrics.get('annual_return', 0):.2%}"
                    ],
                    ['Total Trades', 'Win Rate',
                     'Profit Factor', 'Avg Trade',
                     'Best Trade', 'Worst Trade'],
                    [
                        f"{metrics.get('total_trades', 0)}",
                        f"{metrics.get('win_rate', 0):.2%}",
                        f"{metrics.get('profit_factor', 0):.2f}",
                        f"${metrics.get('avg_trade_pnl', 0):.2f}",
                        (f"${metrics.get('avg_win', 0):.2f}"
                         if 'avg_win' in metrics else 'N/A'),
                        (f"${metrics.get('avg_loss', 0):.2f}"
                         if 'avg_loss' in metrics else 'N/A')
                    ]
                ],
                align=['left', 'right', 'left', 'right'],
                fill_color=['whitesmoke', 'white',
                            'whitesmoke', 'white'],
                font=dict(size=12)
            )
        )

        # Create metrics table (second instance, same content)
        metrics_table = go.Table(
            header=dict(
                values=['Metric', 'Value',
                        'Trade Metrics', 'Value'],
                align=['left', 'right', 'left', 'right'],
                fill_color='royalblue',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    ['Initial Capital', 'Final Value',
                     'Total Return', 'Max Drawdown',
                     'Sharpe Ratio', 'Annual Return'],
                    [
                        f"${metrics.get('initial_value', 0):,.2f}",
                        f"${metrics.get('final_value', 0):,.2f}",
                        f"{metrics.get('total_return', 0):.2%}",
                        f"${metrics.get('max_drawdown_money', 0):,.2f}",
                        f"{metrics.get('sharpe_ratio', 0):.2f}",
                        f"{metrics.get('annual_return', 0):.2%}"
                    ],
                    ['Total Trades', 'Win Rate',
                     'Profit Factor', 'Avg Trade',
                     'Best Trade', 'Worst Trade'],
                    [
                        f"{metrics.get('total_trades', 0)}",
                        f"{metrics.get('win_rate', 0):.2%}",
                        f"{metrics.get('profit_factor', 0):.2f}",
                        f"${metrics.get('avg_trade_pnl', 0):.2f}",
                        (f"${metrics.get('avg_win', 0):.2f}"
                         if 'avg_win' in metrics else 'N/A'),
                        (f"${metrics.get('avg_loss', 0):.2f}"
                         if 'avg_loss' in metrics else 'N/A')
                    ]
                ],
                align=['left', 'right', 'left', 'right'],
                fill_color=['whitesmoke', 'white',
                            'whitesmoke', 'white'],
                font=dict(size=12)
            )
        )

        # Save the Backtrader-style chart as interactive HTML
        backtrader_plot_file = os.path.join(
            output_dir,
            f"{strategy_name}_backtrader_plot.html")
        fig.write_html(
            backtrader_plot_file, config={'responsive': True})
        print(f"Saved Backtrader-style Plotly plot to "
              f"{backtrader_plot_file}")

        # Create individual ticker plots (second pass)
        _create_individual_ticker_plots_second_pass(
            ticker_data, strategy_name, trade_log,
            highlight_trades, ticker_plots_dir)

        # Create separate dashboard with metrics table
        dashboard = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            specs=[
                [{'type': 'xy'}],
                [{'type': 'table'}]
            ],
            subplot_titles=(
                f'{strategy_name}: Dashboard',
                'Performance Metrics'
            ),
            vertical_spacing=0.12
        )

        dashboard.add_trace(
            go.Scatter(
                x=equity_curve_data['dates'],
                y=equity_curve_data['values'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        dashboard.add_trace(
            go.Scatter(
                x=equity_curve_data['dates'],
                y=ec_drawdowns,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1.5),
                yaxis='y2'
            ),
            row=1, col=1
        )

        dashboard.add_trace(metrics_table, row=2, col=1)

        dashboard.update_layout(
            width=1200,
            height=900,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(title='Equity Value',
                       tickprefix='$'),
            yaxis2=dict(
                title='Drawdown',
                tickformat='%',
                overlaying='y',
                side='right',
                range=[min(ec_drawdowns.min() * 1.1, -0.05),
                       0.05]
            )
        )

        # Save the dashboard as interactive HTML
        dashboard_file = os.path.join(
            output_dir,
            f"{strategy_name}_backtest_dashboard.html")
        dashboard.write_html(
            dashboard_file, config={'responsive': True})
        print(f"Saved interactive Plotly dashboard to "
              f"{dashboard_file}")

        # Save static image versions if possible
        if KALEIDO_AVAILABLE:
            try:
                for trace in fig.data:
                    if (isinstance(trace, go.Scatter)
                            and trace.mode
                            and 'markers' in trace.mode):
                        if trace.marker:
                            pass
                        if trace.line:
                            trace.line.width = 3.5
                backtrader_img = os.path.join(
                    output_dir,
                    f"{strategy_name}_backtrader_plot.png")
                fig.write_image(
                    backtrader_img, width=1200,
                    height=fig_height, scale=2)
                print(f"Saved Backtrader-style plot image to "
                      f"{backtrader_img}")

                for trace in dashboard.data:
                    if (isinstance(trace, go.Scatter)
                            and trace.mode
                            and 'markers' in trace.mode):
                        if trace.marker:
                            pass
                        if trace.line:
                            trace.line.width = 3.5
                dashboard_img = os.path.join(
                    output_dir,
                    f"{strategy_name}_backtest_dashboard.png")
                dashboard.write_image(
                    dashboard_img, width=1200,
                    height=900, scale=2)
                print(f"Saved dashboard static image to "
                      f"{dashboard_img}")
            except Exception as img_error:
                print(f"Warning: Static images could not be "
                      f"saved: {img_error}")
        else:
            print("Note: Static image export is disabled. "
                  "To enable, install kaleido package with "
                  "'pip install kaleido'.")
    else:
        # Fallback to backtrader's plotting if no equity curve
        print("No equity curve data available, using "
              "backtrader's native plotting")
        fig = cerebro.plot(
            style='candle', barup='green', bardown='red',
            volume=False, grid=True)
        plot_file = os.path.join(
            output_dir,
            f"{strategy_name}_backtest_plot.png")
        if isinstance(fig, list) and len(fig) > 0:
            fig[0][0].savefig(plot_file)
            print(f"Saved plot to {plot_file}")
        elif hasattr(fig, 'savefig'):
            fig.savefig(plot_file)
            print(f"Saved plot to {plot_file}")
