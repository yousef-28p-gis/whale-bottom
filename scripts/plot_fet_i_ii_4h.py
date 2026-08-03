"""FET 4H — Waves i & ii: ATL → wave i → wave ii with zigzag"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_fet_4h_w1w2.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

# Focus: March 10 to April 5
mask = (df['dt'] >= '2020-03-10') & (df['dt'] <= '2020-04-05')
df = df[mask].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

# Zigzag on 4h data
zz = zigzag(highs_arr, lows_arr, depth=8, dev=3.0)

fig, ax = plt.subplots(figsize=(26, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Candles with wicks
for i, row in df.iterrows():
    c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    x = mdates.date2num(row['dt'])
    bb = min(row['open'], row['close'])
    bh = max(abs(row['close']-row['open']), 0.00002)
    w = 0.15
    # Wick (high-low line)
    ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.7, zorder=1)
    # Body
    ax.add_patch(plt.Rectangle((x-w/2, bb), w, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.2, zorder=2))

# Find exact ATL from data
atl_idx = lows_arr.argmin()
atl_price = lows_arr[atl_idx]
atl_dt = df['dt'].iloc[atl_idx]

# Find exact wave i peak and wave ii bottom
wi_idx = None; wii_idx = None
for i in range(len(df)):
    if df['dt'].iloc[i] == pd.Timestamp('2020-03-27 16:00'):
        wi_idx = i
    if df['dt'].iloc[i] == pd.Timestamp('2020-03-30 00:00'):
        wii_idx = i

if wi_idx is None:
    # Fallback: find nearest high to 0.0157 after ATL
    for i in range(atl_idx, len(df)):
        if abs(highs_arr[i] - 0.0157) < 0.0005:
            wi_idx = i; break
if wii_idx is None:
    # Fallback: find nearest low to 0.0124 after w1
    for i in range(wi_idx, len(df)):
        if abs(lows_arr[i] - 0.0124) < 0.0005:
            wii_idx = i; break

wi_price = highs_arr[wi_idx]
wi_dt = df['dt'].iloc[wi_idx]
wii_price = lows_arr[wii_idx]
wii_dt = df['dt'].iloc[wii_idx]

print(f'ATL: {atl_price:.6f} at {atl_dt}')
print(f'Wave i: {wi_price:.6f} at {wi_dt}')
print(f'Wave ii: {wii_price:.6f} at {wii_dt}')

# Main wave i & ii lines
ax.plot([atl_dt, wi_dt], [atl_price, wi_price], color='#2196F3', linewidth=3, zorder=4, alpha=0.85)
ax.plot([wi_dt, wii_dt], [wi_price, wii_price], color='#FF5722', linewidth=3, zorder=4, alpha=0.85)

# Main labels
for d, p, label, color in [
    (atl_dt, atl_price, '0', '#333'),
    (wi_dt, wi_price, 'i', '#2196F3'),
    (wii_dt, wii_price, 'ii', '#FF5722'),
]:
    va = 'top' if label == 'i' else 'bottom'
    yoff = -15 if va == 'top' else 15
    ax.annotate(label, (d, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=14, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.95), zorder=10)

# ---- Zigzag ----
for idx in range(len(zz)-1):
    i1, p1, t1 = zz[idx]
    i2, p2, t2 = zz[idx+1]
    d1, d2 = df['dt'].iloc[i1], df['dt'].iloc[i2]
    ax.plot([d1, d2], [p1, p2], color='#888', linewidth=1.0, linestyle='--', zorder=3, alpha=0.6)

for i, p, t in zz:
    d = df['dt'].iloc[i]
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=25, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=0.5)

# ---- Fibonacci ----
i_len = wi_price - atl_price
ii_ret = (wi_price - wii_price) / i_len * 100

# Fib retracement levels inside wave i
for f in [0.382, 0.5, 0.618, 0.786]:
    price = wi_price - (i_len * f)
    ax.axhline(y=price, color='#bbb', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.annotate(f'{f:.3f}', (pd.Timestamp('2020-04-04'), price),
                fontsize=6.5, color='#999', ha='left', va='center')

# ii retracement label
ax.annotate(f'ii = {ii_ret:.1f}% of i\n≈ Fib 0.382', xy=(wii_dt, wii_price),
            xytext=(40, 22), textcoords="offset points",
            fontsize=10, color='#FF5722', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.3),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#FF5722', alpha=0.85))

# Wave i length
ax.annotate(f'i = {i_len:.4f}\n(+{i_len/0.0074*100:.0f}%)',
            xy=(pd.Timestamp('2020-03-20'), 0.0110),
            fontsize=10, color='#2196F3', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2196F3', alpha=0.85))

# ---- Axes ----
ax.set_yscale('log')
ax.set_xlim(df['dt'].min(), pd.Timestamp('2020-04-05'))
ax.set_ylim(0.006, 0.020)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.4f}'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_title('FET/USDT — Waves i & ii on 4H\n'
             f'ATL→i→ii | i=${0.0157:.4f} (+{i_len/0.0074*100:.0f}%) | '
             f'ii retrace = {ii_ret:.1f}% (Fib 0.382)',
             fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
out = '/data/trading28/charts/fet_i_ii_4h.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
print(f'Wave i:  {0.0074:.4f} → {0.0157:.4f}  len={i_len:.4f} (+{i_len/0.0074*100:.0f}%)')
print(f'Wave ii: {0.0157:.4f} → {0.0124:.4f}  retrace={ii_ret:.1f}%')

# Show zz pivots
print(f'\nZigzag pivots ({len(zz)}):')
for i, p, t in zz:
    dt_str = df['dt'].iloc[i].strftime('%m-%d %H:%M')
    print(f'  {dt_str}  {t}: {p:.4f}')
