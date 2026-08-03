#!/usr/bin/env python3
"""FET 15m + Whale Indicator — Plot"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

ex = ccxt.binance({'timeout': 15000})
since = ex.parse8601((datetime.utcnow() - timedelta(days=30)).isoformat())
all_c = []
while True:
    batch = ex.fetch_ohlcv('FET/USDT', '15m', since=since, limit=1000)
    if not batch: break
    all_c.extend(batch)
    since = batch[-1][0] + 1
    if len(batch) < 1000: break

df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True); df.sort_index(inplace=True)

c = df['close'].values; l = df['low'].values
h = df['high'].values; v = df['volume'].values
n = len(c)

# ── Whale Indicator ──
lookback = 200
low_change = np.zeros(n)
for i in range(1,n): low_change[i] = abs(l[i]-l[i-1])/l[i]*100
smooth_change = pd.Series(low_change).ewm(span=3,adjust=False).mean().values
lowest_n = pd.Series(l).rolling(lookback).min().values
highest_change = pd.Series(smooth_change).rolling(lookback).max().values
strength_raw = np.zeros(n)
for i in range(lookback,n):
    if l[i] <= lowest_n[i]:
        strength_raw[i] = (smooth_change[i] + highest_change[i]*2)/3
whale = pd.Series(strength_raw).ewm(span=3,adjust=False).mean().values
wma20 = pd.Series(whale).rolling(20).apply(lambda x: np.average(x, weights=np.arange(1,21))).values
wma50 = pd.Series(whale).rolling(50).apply(lambda x: np.average(x, weights=np.arange(1,51))).values
highest_whale_50 = pd.Series(whale).rolling(50).max().values
whale_str = np.zeros(n)
for i in range(50,n):
    if highest_whale_50[i]>0: whale_str[i] = whale[i]/highest_whale_50[i]*100
whale_spike = np.zeros(n,bool)
for i in range(1,n):
    if whale[i] > whale[i-1] and whale[i-1] <= 0.02: whale_spike[i]=True

# ── Other indicators ──
ema21 = pd.Series(c).ewm(span=21,adjust=False).mean().values
ema50 = pd.Series(c).ewm(span=50,adjust=False).mean().values

# ── Chart ──
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.03,
    subplot_titles=('FET/USDT 15m + مؤشر الحوت', 'Whale Strength %', 'Whale + WMA20/50'))

# Candles
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
    low=df['low'], close=df['close'], name='FET', showlegend=False), row=1, col=1)

# EMA21
fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA21', line=dict(color='orange',width=1)), row=1, col=1)

# Whale spike signals
spike_idx = df.index[whale_spike]
spike_px = c[whale_spike]
fig.add_trace(go.Scatter(x=spike_idx, y=spike_px, mode='markers',
    marker=dict(symbol='triangle-up',size=10,color='cyan',line=dict(color='blue',width=1)),
    name='🐋 Whale Spike'), row=1, col=1)

# Whale Strength %
fig.add_trace(go.Scatter(x=df.index, y=whale_str, name='Strength%',
    line=dict(color='purple',width=1)), row=2, col=1)
fig.add_hline(y=50, line_dash='dash', line_color='gray', row=2, col=1)

# Whale + WMAs
fig.add_trace(go.Scatter(x=df.index, y=whale, name='Whale Raw',
    line=dict(color='cyan',width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=wma20, name='WMA20',
    line=dict(color='lime',width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=wma50, name='WMA50',
    line=dict(color='red',width=1)), row=3, col=1)

fig.update_layout(
    title='🐋 مؤشر الحوت الخام — FET/USDT 15m',
    template='plotly_white',
    height=900,
    hovermode='x unified',
    xaxis_rangeslider_visible=False,
)

fig.update_yaxes(title_text='Price', row=1, col=1)
fig.update_yaxes(title_text='%', row=2, col=1)
fig.update_yaxes(title_text='Whale', row=3, col=1)

fig.write_html('/data/trading28/charts/fet_whale_raw.html')
print(f'✅ Saved: /data/trading28/charts/fet_whale_raw.html')
print(f'Candles: {n}, Spikes: {whale_spike.sum()}')
print(f'Last whale: {whale[-1]:.4f}, Str: {whale_str[-1]:.0f}%')
