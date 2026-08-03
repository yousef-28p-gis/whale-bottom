"""BTC 4H — Elliott Wave: ATL Nov 2022 to Oct 2023"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_btc_15m_nov22.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

mask = (df['dt'] >= '2022-11-21') & (df['dt'] <= '2022-12-05')
df = df[mask].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

zz = zigzag(highs_arr, lows_arr, depth=8, dev=3.0)

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

for idx in range(len(zz)-1):
    i1, p1, t1 = zz[idx]
    i2, p2, t2 = zz[idx+1]
    d1, d2 = df['dt'].iloc[i1], df['dt'].iloc[i2]
    ax.plot([d1, d2], [p1, p2], color='#FF5722', linewidth=1.5, zorder=4, alpha=0.9)

for i, p, t in zz:
    d = df['dt'].iloc[i]
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=25, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=0.5)

atl_idx = lows_arr.argmin()
atl_price = lows_arr[atl_idx]
atl_dt = df['dt'].iloc[atl_idx]
ax.scatter(atl_dt, atl_price, s=120, c='red', marker='v', zorder=10, edgecolors='white', linewidths=1.5)
ax.annotate('ATL $' + f'{atl_price:,.0f}', xy=(atl_dt, atl_price),
            xytext=(0, -12), textcoords="offset points",
            fontsize=10, fontweight='bold', color='red', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.95), zorder=10)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_title('BTC/USDT — Elliott Wave from ATL: Nov 21 - Dec 5, 2022\n'
             '15m Timeframe | Zigzag D=8 dev=3%',
             fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
out = '/data/trading28/charts/btc_elliott_4h.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: ' + out)
for i, p, t in zz:
    dt_str = df['dt'].iloc[i].strftime('%Y-%m-%d')
    print('  ' + dt_str + '  ' + t + ': ' + f'{p:,.0f}')
