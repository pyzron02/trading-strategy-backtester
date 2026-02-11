#!/usr/bin/env python3
"""Shared visualization utilities, color schemes, and layout templates."""

# Standard color palette used across all visualizations
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'danger': '#d62728',       # Red
    'warning': '#bcbd22',      # Yellow-green
    'info': '#17becf',         # Cyan
    'equity': '#1f77b4',       # Equity curve
    'drawdown': '#d62728',     # Drawdown
    'benchmark': '#7f7f7f',    # Benchmark/reference
    'buy': '#00cc00',          # Buy signals
    'sell': '#cc0000',         # Sell signals
}

# Plotly layout defaults
PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Arial, sans-serif', size=12),
    title_font_size=16,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=60, r=40, t=60, b=40),
)

# Matplotlib style settings
MPL_STYLE = {
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

def get_plotly_layout(**overrides):
    """Get a copy of the default Plotly layout with optional overrides."""
    layout = PLOTLY_LAYOUT.copy()
    layout.update(overrides)
    return layout

def safe_save_plotly_html(fig, filepath, include_plotlyjs='cdn'):
    """Save a Plotly figure to HTML with error handling."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        fig.write_html(filepath, include_plotlyjs=include_plotlyjs)
        return True
    except Exception as e:
        print(f"Warning: Could not save Plotly HTML to {filepath}: {e}")
        return False

def safe_save_plotly_png(fig, filepath, width=1200, height=800, scale=2):
    """Save a Plotly figure to PNG with kaleido/orca fallback."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        fig.write_image(filepath, width=width, height=height, scale=scale)
        return True
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido may not be installed): {e}")
        return False
