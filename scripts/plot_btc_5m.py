"""BTC 5m — Elliott Wave: ATL $15,476 Nov 2022"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_btc_5m_nov2022.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

mask = (df['dt'] >= '2022-11-21') & (df['dt'] <= '2022-12-01')
df = df[mask].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

zz = zigzag(highs_arr, lows_arr, depth=5, dev=2.0)

fig, ax = plt.subplots(figsize=(28, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

for i, row in df.iterrows():
    c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    x = mdates.date2num(row['dt'])
    bb = min(row['open'], row['close'])
    bh = max(abs(row['close']-row['open']), 10.0)
    w = 0.15
    ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.3, zorder=1, alpha=0.6)
    ax.add_patch(plt.Rectangle((x-w/2, bb), w, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.1, zorder=2))

# ZigZag lines
for idx in range(len(zz)-1):
    i1, p1, t1 = zz[idx]
    i2, p2, t2 = zz[idx+1]
    d1, d2 = df['dt'].iloc[i1], df['dt'].iloc[i2]
    ax.plot([d1, d2], [p1, p2], color='#FF5722', linewidth=1.5, zorder=4, alpha=0.9)

# ZigZag points
for i, p, t in zz:
    d = df['dt'].iloc[i]
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=25, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=0.5)

atl_idx = lows_arr.argmin()
atl_price = lows_arr[atl_idx]
atl_dt = df['dt'].iloc[atl_idx]

# Label bottom
ax.annotate(f'ATL\n${atl_price:,.0f}', xy=(atl_dt, atl_price),
            xytext=(atl_dt, atl_price - 200), fontsize=12, fontweight='bold',
            color='#E91E63', ha='center',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))

ax.set_title('BTC/USDT — 5 دقائق من القاع التاريخي (21-30 نوفمبر 2022)', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('السعر (USDT)', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('/data/trading28/charts/btc_5m_nov2022.png', dpi=150, facecolor='white')
print("Chart saved: btc_5m_nov2022.png")
print(f"ZigZag pivots: {len(zz)}")
for i, p, t in zz:
    print(f"  {df['dt'].iloc[i]} | {t} | ${p:,.0f}")
