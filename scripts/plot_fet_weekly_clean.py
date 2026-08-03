"""FET Weekly — Clean chart, no labeling, ready for fresh count"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open('/data/trading28/data_fet_weekly.json') as f:
    candles = json.load(f)
df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values

hist_low_idx = lows.argmin()
hist_low_price = lows[hist_low_idx]
hist_low_dt = df['dt'].iloc[hist_low_idx]
hist_high_idx = highs.argmax()
hist_high_price = highs[hist_high_idx]
hist_high_dt = df['dt'].iloc[hist_high_idx]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 16), gridspec_kw={'height_ratios':[3.5,1]}, sharex=True)
fig.patch.set_facecolor('white')

# Candles
for i in range(len(df)):
    c = '#26a69a' if closes[i] >= df['open'].iloc[i] else '#ef5350'
    bb = min(df['open'].iloc[i], closes[i])
    bh = max(abs(closes[i]-df['open'].iloc[i]), 0.0001)
    ax1.add_patch(plt.Rectangle((mdates.date2num(df['dt'].iloc[i])-2.5, bb), 5, bh,
                               facecolor=c, edgecolor=c, linewidth=0.5))
    ax1.plot([mdates.date2num(df['dt'].iloc[i]), mdates.date2num(df['dt'].iloc[i])],
            [df['low'].iloc[i], df['high'].iloc[i]], color=c, linewidth=0.8)

# Zigzag
pivots = zigzag(highs, lows, depth=6, dev=1.0)
zz_x = [df['dt'].iloc[b] for b,_,_ in pivots]
zz_y = [p for _,p,_ in pivots]
ax1.plot(zz_x, zz_y, color='#1565C0', linewidth=2.2, zorder=5)

# Pivot dots only — no labels
for pi, (idx, price, ptype) in enumerate(pivots):
    dt = df['dt'].iloc[idx]
    color = '#E91E63' if ptype == 'H' else '#2196F3'
    marker = 'v' if ptype == 'H' else '^'
    ax1.scatter(dt, price, s=60, c=color, zorder=6, edgecolors='white', linewidth=1.5, marker=marker)

# ATL & ATH
ax1.scatter(hist_low_dt, hist_low_price, marker='v', s=300, c='#C62828', zorder=10, edgecolors='white', linewidth=3)
ax1.annotate(f'ATL ${hist_low_price:.5f}', (hist_low_dt, hist_low_price),
            textcoords="offset points", xytext=(0, -40), ha='center',
            fontsize=10, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#C62828', alpha=0.9))
ax1.scatter(hist_high_dt, hist_high_price, marker='^', s=300, c='#2E7D32', zorder=10, edgecolors='white', linewidth=3)
ax1.annotate(f'ATH ${hist_high_price:.2f}', (hist_high_dt, hist_high_price),
            textcoords="offset points", xytext=(0, 30), ha='center',
            fontsize=10, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2E7D32', alpha=0.9))

ax1.set_yscale('log')
ax1.set_facecolor('white')
ax1.grid(True, alpha=0.25, color='#e0e0e0')
ax1.set_title('FET/USDT Weekly — Zigzag (depth=6, dev=1.0%) | Clean — No Labels', 
              fontsize=15, color='#212121', fontweight='bold')
ax1.set_ylabel('Price (USDT) — Log', fontsize=11, color='#555')
for sp in ax1.spines.values():
    sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)

# Volume
vc = ['#ef5350' if closes[i] < df['open'].iloc[i] else '#26a69a' for i in range(len(df))]
ax2.bar(df['dt'], df['vol'], color=vc, alpha=0.4, width=3)
ax2.set_ylabel('Volume', fontsize=11, color='#555')
ax2.grid(True, alpha=0.3, color='#e0e0e0')
ax2.set_facecolor('white')
for sp in ax2.spines.values():
    sp.set_edgecolor('#ddd'); sp.set_linewidth(0.5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
fig.savefig('/data/trading28/scripts/fet_weekly_zigzag.png', dpi=150, facecolor='white', bbox_inches='tight')
print(f'Done — {len(pivots)} pivots')
for i, (idx, price, ptype) in enumerate(pivots):
    dt = df['dt'].iloc[idx]
    print(f'P{i}={ptype} {dt.strftime("%Y-%m-%d")} ${price:.4f}')
