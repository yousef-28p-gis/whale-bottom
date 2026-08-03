#!/usr/bin/env python3
"""FET 7d + Whale — matplotlib"""
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

c=df['close'].values; l=df['low'].values; h=df['high'].values; n=len(c)
idx=df.index

lookback=200
low_change=np.zeros(n)
for i in range(1,n): low_change[i]=abs(l[i]-l[i-1])/l[i]*100
sc=pd.Series(low_change).ewm(span=3,adjust=False).mean().values
ln=pd.Series(l).rolling(lookback).min().values
hc=pd.Series(sc).rolling(lookback).max().values
sr=np.zeros(n)
for i in range(lookback,n):
    if l[i]<=ln[i]: sr[i]=(sc[i]+hc[i]*2)/3
whale=pd.Series(sr).ewm(span=3,adjust=False).mean().values
w20=pd.Series(whale).rolling(20).apply(lambda x: np.average(x,weights=np.arange(1,21))).values
w50=pd.Series(whale).rolling(50).apply(lambda x: np.average(x,weights=np.arange(1,51))).values
hw=pd.Series(whale).rolling(50).max().values
ws=np.zeros(n)
for i in range(50,n):
    if hw[i]>0: ws[i]=whale[i]/hw[i]*100
spike=np.zeros(n,bool)
for i in range(1,n):
    if whale[i]>whale[i-1] and whale[i-1]<=0.02: spike[i]=True

ema21=pd.Series(c).ewm(span=21,adjust=False).mean().values

# ── Plot ──
plt.style.use('default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
    gridspec_kw={'height_ratios': [2, 1, 1]})
fig.patch.set_facecolor('white')

# Chart 1: Price + EMA21 + spikes
ax1.set_facecolor('white')
colors = ['green' if c[i] >= df['open'].values[i] else 'red' for i in range(n)]
ax1.bar(idx, h-l, bottom=l, color=colors, width=0.0004, alpha=0.3, linewidth=0)
ax1.bar(idx, abs(c-df['open'].values), bottom=np.minimum(c,df['open'].values), color=colors, width=0.0003, linewidth=0)
ax1.plot(idx, ema21, 'orange', linewidth=1, label='EMA21')
ax1.plot(idx[spike], c[spike], 'c^', markersize=10, label=f'🐋 ({spike.sum()})', zorder=5)
ax1.set_ylabel('FET/USDT'); ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Chart 2: Whale Strength %
ax2.set_facecolor('white')
ax2.plot(idx, ws, 'purple', linewidth=1)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(idx, 50, ws, where=ws>50, color='green', alpha=0.15)
ax2.set_ylabel('Strength %'); ax2.grid(True, alpha=0.3)

# Chart 3: Whale + WMAs
ax3.set_facecolor('white')
ax3.plot(idx, whale, 'cyan', linewidth=1.5, label='Whale')
ax3.plot(idx, w20, 'lime', linewidth=1, label='WMA20')
ax3.plot(idx, w50, 'red', linewidth=1, label='WMA50')
ax3.set_ylabel('Whale'); ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
fig.suptitle(f'🐋 مؤشر الحوت الخام — FET/USDT 15m — 7 أيام ({spike.sum()} إشارة)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/data/trading28/charts/fet_whale_7d.png', dpi=120, facecolor='white', bbox_inches='tight')
plt.close()
print(f'✅ Saved: fet_whale_7d.png | {n} candles | {spike.sum()} spikes')
