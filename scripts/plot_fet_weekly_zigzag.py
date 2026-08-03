"""
FET Weekly chart with ZigZag indicator — white background
"""
import json
import sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Load data
with open('/data/trading28/data_fet_weekly.json') as f:
    candles = json.load(f)

# Parse
df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

highs = df['high'].values
lows = df['low'].values
closes = df['close'].values

# ZigZag
pivots = zigzag(highs, lows, depth=10, dev=1.0)

# Build zigzag lines
zz_x = []
zz_y = []
for i, (bar_idx, price, ptype) in enumerate(pivots):
    zz_x.append(df['dt'].iloc[bar_idx])
    zz_y.append(price)

# Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.03)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df['dt'],
    open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name='FET/USDT',
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
), row=1, col=1)

# ZigZag line
fig.add_trace(go.Scatter(
    x=zz_x, y=zz_y,
    mode='lines+markers',
    line=dict(color='#1565C0', width=2.5),
    marker=dict(size=8, color='#1565C0'),
    name='ZigZag',
), row=1, col=1)

# Volume
colors = ['#ef5350' if df['close'].iloc[i] < df['open'].iloc[i] else '#26a69a' for i in range(len(df))]
fig.add_trace(go.Bar(
    x=df['dt'], y=df['vol'],
    name='Volume',
    marker_color=colors,
    opacity=0.4,
), row=2, col=1)

# Layout — white background
fig.update_layout(
    title=dict(text='FET/USDT — Weekly + ZigZag (depth=10, dev=1.0%)', x=0.5, font=dict(size=18, color='#212121')),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#212121'),
    xaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
    xaxis2=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
    yaxis2=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    hovermode='x unified',
    height=800,
)

fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)
fig.update_yaxes(title_text='Volume', row=2, col=1)

fig.write_html('/data/trading28/scripts/fet_weekly_zigzag.html')
print(f'Done: {len(pivots)} pivots, {len(candles)} candles')
print(f'Chart: /data/trading28/scripts/fet_weekly_zigzag.html')
