#!/usr/bin/env python3
"""Whale Pump Simple — Pine Script port — FET 15m"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings; warnings.filterwarnings('ignore')

ex = ccxt.binance({'timeout': 15000})
since = ex.parse8601((datetime.utcnow() - timedelta(days=7)).isoformat())
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

c=df['close'].values; l=df['low'].values; h=df['high'].values; idx=df.index; n=len(c)

# ── Whale Pump Simple ──
LB = 30
ln = pd.Series(l).rolling(LB).min().values  # lowest_30
at_low = l <= ln

low_change = np.zeros(n)
for i in range(1,n): low_change[i] = abs(l[i]-l[i-1])/l[i]*100
sc = pd.Series(low_change).ewm(span=3,adjust=False).mean().values
hc = pd.Series(sc).rolling(LB).max().values

strength = np.where(at_low, (sc + hc*2)/3, 0)
whale_pump = pd.Series(strength).ewm(span=3,adjust=False).mean().values

up = whale_pump > np.roll(whale_pump, 1)

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]})
fig.patch.set_facecolor('white')

# Price
ax1.set_facecolor('white')
colors = ['green' if c[i] >= df['open'].values[i] else 'red' for i in range(n)]
ax1.bar(idx, h-l, bottom=l, color=colors, width=0.0004, alpha=0.3)
ax1.bar(idx, abs(c-df['open'].values), bottom=np.minimum(c,df['open'].values), color=colors, width=0.0003)
ax1.set_ylabel('FET/USDT')
ax1.grid(True, alpha=0.3)

# Whale Pump columns
ax2.set_facecolor('white')
green_bars = up & (whale_pump > 0)
red_bars = ~up & (whale_pump > 0)
ax2.bar(idx[green_bars], whale_pump[green_bars], width=0.0007, color='limegreen', alpha=0.8)
ax2.bar(idx[red_bars], whale_pump[red_bars], width=0.0007, color='red', alpha=0.8)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.set_ylabel('Whale Pump'); ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
fig.suptitle('🐋 Whale Pump Simple — FET/USDT 15m — 7 أيام', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/fet_whale_simple.png', dpi=120, facecolor='white', bbox_inches='tight')
plt.close()

# Stats
signals = whale_pump > whale_pump[50:].mean()*3 if n>50 else whale_pump > 0.01
print(f'✅ fet_whale_simple.png | {n} candles | Peak whale: {whale_pump.max():.4f}')
print(f'   Bars with signal (>mean*3): {signals.sum()}/{n}')
print(f'   Green bars: {green_bars.sum()}, Red bars: {red_bars.sum()}')
