"""FET 1m — Wave 1: ATL → W1 peak with zigzag"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_fet_1m_w1.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

# Focus: 02:00 to 05:05
mask = (df['dt'] >= '2020-03-13 02:00') & (df['dt'] <= '2020-03-13 05:05')
df = df[mask].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

# Zigzag on 1m data
zz = zigzag(highs_arr, lows_arr, depth=10, dev=3.0)

fig, (ax, ax_macd, ax_rsi) = plt.subplots(3, 1, figsize=(22, 14),
    gridspec_kw={'height_ratios': [4, 1.5, 1.5]}, sharex=True)
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Candles with wicks — thinner since 1m
for i, row in df.iterrows():
    c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    x = mdates.date2num(row['dt'])
    bb = min(row['open'], row['close'])
    bh = max(abs(row['close']-row['open']), 0.000003)
    w = 0.00035
    ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.3, zorder=1, alpha=0.6)
    ax.add_patch(plt.Rectangle((x-w/2, bb), w, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.1, zorder=2))

# Zigzag
for idx in range(len(zz)-1):
    i1, p1, t1 = zz[idx]
    i2, p2, t2 = zz[idx+1]
    d1, d2 = df['dt'].iloc[i1], df['dt'].iloc[i2]
    ax.plot([d1, d2], [p1, p2], color='#FF5722', linewidth=1.5, zorder=4, alpha=0.9)

for i, p, t in zz:
    d = df['dt'].iloc[i]
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=30, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=0.8)

# ATL & W1 markers
atl_idx = lows_arr.argmin()
atl_price = lows_arr[atl_idx]
atl_dt = df['dt'].iloc[atl_idx]

w1_idx = highs_arr.argmax()
w1_price = highs_arr[w1_idx]
w1_dt = df['dt'].iloc[w1_idx]

# Main wave line
ax.plot([atl_dt, w1_dt], [atl_price, w1_price], color='#2196F3', linewidth=2.5, zorder=3, alpha=0.7)

# Labels
ax.annotate('ATL\n$' + f'{atl_price:.5f}', xy=(atl_dt, atl_price),
            xytext=(0, -10), textcoords="offset points",
            fontsize=10, fontweight='bold', color='red', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.95), zorder=10)
ax.scatter(atl_dt, atl_price, s=120, c='red', marker='v', zorder=10,
           edgecolors='white', linewidths=1.5)

ax.annotate('W1\n$' + f'{w1_price:.5f}', xy=(w1_dt, w1_price),
            xytext=(0, -8), textcoords="offset points",
            fontsize=10, fontweight='bold', color='#2196F3', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2196F3', alpha=0.95), zorder=10)

# ---- Subwave labels: placed on key structural pivots ----
# Zigzag D=10 dev=3% shows full detail; labels at main 5-wave structure
# zz[4]=02:45 H:0.01124 (W1), zz[5]=02:56 L:0.00949 (W2)
# zz[6]=03:28 H:0.01215 (W3), zz[15]=04:58 L:0.01001 (W4)
if len(zz) >= 16:
    label_map = [
        (4, '1', '#2196F3', 'top'),
        (5, '2', '#FF5722', 'bottom'),
        (6, '3', '#2196F3', 'top'),
        (15, '4', '#FF5722', 'bottom'),
    ]
    
    for zi, label, color, va in label_map:
        i, p, t = zz[zi]
        d = df['dt'].iloc[i]
        yoff = -8 if va == 'top' else 8
        ax.annotate(label, (d, p), textcoords="offset points", xytext=(0, yoff),
                    fontsize=12, fontweight='bold', color=color, ha='center', va=va,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.95), zorder=10)
    
    i5, p5, t5 = zz[15]
    ax.annotate('5 →', (df['dt'].iloc[i5], p5), textcoords="offset points",
                xytext=(20, 0), fontsize=12, fontweight='bold', color='#2196F3', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2196F3', alpha=0.95), zorder=10)

# ---- MACD ----
close_arr = df['close'].values
ema12 = pd.Series(close_arr).ewm(span=12, adjust=False).mean().values
ema26 = pd.Series(close_arr).ewm(span=26, adjust=False).mean().values
macd_line = ema12 - ema26
signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
macd_hist = macd_line - signal_line

ax_macd.set_facecolor('#fafafa')
ax_macd.fill_between(df['dt'], 0, macd_hist,
                      where=macd_hist >= 0, color='#26a69a', alpha=0.6, linewidth=0)
ax_macd.fill_between(df['dt'], 0, macd_hist,
                      where=macd_hist < 0, color='#ef5350', alpha=0.6, linewidth=0)
ax_macd.plot(df['dt'], macd_line, color='#2196F3', linewidth=1, label='MACD')
ax_macd.plot(df['dt'], signal_line, color='#FF9800', linewidth=0.8, label='Signal')
ax_macd.axhline(y=0, color='#999', linewidth=0.5, linestyle='--')
ax_macd.grid(True, alpha=0.3, linestyle='--')
ax_macd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.6f}'))
ax_macd.set_ylabel('MACD', fontsize=9, fontweight='bold')
ax_macd.legend(loc='upper left', fontsize=7, ncol=2)

# ---- RSI ----
delta = pd.Series(close_arr).diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

ax_rsi.set_facecolor('#fafafa')
ax_rsi.plot(df['dt'], rsi, color='#9C27B0', linewidth=1.2)
ax_rsi.axhline(y=70, color='#ef5350', linewidth=0.5, linestyle='--', alpha=0.6)
ax_rsi.axhline(y=30, color='#26a69a', linewidth=0.5, linestyle='--', alpha=0.6)
ax_rsi.axhline(y=50, color='#999', linewidth=0.5, linestyle='-', alpha=0.3)
ax_rsi.fill_between(df['dt'], 70, 100, color='#ef5350', alpha=0.08)
ax_rsi.fill_between(df['dt'], 0, 30, color='#26a69a', alpha=0.08)
ax_rsi.set_ylim(0, 100)
ax_rsi.grid(True, alpha=0.3, linestyle='--')
ax_rsi.set_ylabel('RSI(14)', fontsize=9, fontweight='bold')

# ---- Axes format ----
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.5f}'))
ax.grid(True, alpha=0.3, linestyle='--')

ax_rsi.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax.set_title('FET/USDT — Wave 1 Zoom: ATL to 05:02 on 1-Minute | Mar 13, 2020\n'
             + 'ATL $' + f'{atl_price:.5f}' + ' → ' + f'${highs_arr.max():.5f}'
             + ' | Zigzag D=10 dev=3%',
             fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
out = '/data/trading28/charts/fet_w1_1m.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: ' + out)
print('ATL: ' + f'{atl_price:.5f}' + ' at ' + atl_dt.strftime('%H:%M'))
w1_peak_price = highs_arr.max()
print('Peak in view: ' + f'{w1_peak_price:.5f}')
print('Zigzag pivots: ' + str(len(zz)))
for i, p, t in zz:
    print('  ' + df['dt'].iloc[i].strftime('%H:%M') + '  ' + t + ': ' + f'{p:.5f}')
