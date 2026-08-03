"""FET 15m — Mar 13-16 with Zigzag"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_fet_15m.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')
df['date'] = df['dt'].dt.date

# Mar 13 to Mar 16
target_dates = [
    pd.Timestamp('2020-03-13').date(),
    pd.Timestamp('2020-03-14').date(),
    pd.Timestamp('2020-03-15').date(),
    pd.Timestamp('2020-03-16').date(),
]
df = df[df['date'].isin(target_dates)].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

# Zigzag — tuned for exactly 2 waves (ATL→W1→W2)
zz = zigzag(highs_arr, lows_arr, depth=8, dev=30.0)

fig, ax = plt.subplots(figsize=(26, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Candles with wicks
for i, row in df.iterrows():
    c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    x = mdates.date2num(row['dt'])
    bb = min(row['open'], row['close'])
    bh = max(abs(row['close']-row['open']), 0.00002)
    w = 0.008
    ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.5, zorder=1)
    ax.add_patch(plt.Rectangle((x-w/2, bb), w, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.1, zorder=2))

# Zigzag lines
for idx in range(len(zz)-1):
    i1, p1, t1 = zz[idx]
    i2, p2, t2 = zz[idx+1]
    d1, d2 = df['dt'].iloc[i1], df['dt'].iloc[i2]
    ax.plot([d1, d2], [p1, p2], color='#FF5722', linewidth=1.5, zorder=4, alpha=0.9)

# Zigzag dots
for i, p, t in zz:
    d = df['dt'].iloc[i]
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=35, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=1)

# ATL marker
atl_row = df[df['low'] == df['low'].min()].iloc[0]
atl_low = atl_row['low']
ax.scatter(atl_row['dt'], atl_low, s=150, c='red', marker='v', zorder=10,
           edgecolors='white', linewidths=2)
ax.annotate('ATL\n$' + f'{atl_low:.5f}', xy=(atl_row['dt'], atl_low),
            xytext=(0, -25), textcoords="offset points",
            fontsize=11, fontweight='bold', color='red', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.95), zorder=10)

# Zigzag labels — just 3 pivots = 2 waves
# W1 at zz[1] (H:0.01403), W2 at zz[2] (L:0.00892)
if len(zz) >= 3:
    # Wave 1 label
    i1, p1, t1 = zz[1]
    d1 = df['dt'].iloc[i1]
    ax.annotate('1', (d1, p1), textcoords="offset points", xytext=(0, -14),
                fontsize=13, fontweight='bold', color='#2196F3', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2196F3', alpha=0.95), zorder=10)
    # Wave 2 label
    i2, p2, t2 = zz[2]
    d2 = df['dt'].iloc[i2]
    ax.annotate('2', (d2, p2), textcoords="offset points", xytext=(0, 14),
                fontsize=13, fontweight='bold', color='#FF5722', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF5722', alpha=0.95), zorder=10)
    
    # W1 length & W2 retrace
    w1_len = p1 - zz[0][1]
    w2_ret = (p1 - p2) / w1_len * 100
    mid_d = d1 + (d2 - d1) / 2
    ax.annotate(f'W1={w1_len:.5f}\nW2 retrace={w2_ret:.1f}%\n≈ Fib 0.382', 
                xy=(mid_d, 0.0095), fontsize=9, fontweight='bold', color='#FF5722', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#FF5722', alpha=0.85), zorder=10)
    
    print(f'W1: {zz[0][1]:.5f} -> {p1:.5f}  len={w1_len:.5f}')
    print(f'W2: {p1:.5f} -> {p2:.5f}  retrace={w2_ret:.1f}%')
for d in target_dates:
    day_start = pd.Timestamp(d)
    ax.axvline(x=day_start, color='#888', linestyle='--', linewidth=0.8, alpha=0.4)
    
    day_df = df[df['date'] == d]
    noon = day_start + pd.Timedelta(hours=12)
    d_low = day_df['low'].min()
    d_high = day_df['high'].max()
    d_open = day_df['open'].iloc[0]
    d_close = day_df['close'].iloc[-1]
    d_range = (d_high - d_low) / d_low * 100
    
    c = '#26a69a' if d_close >= d_open else '#ef5350'
    day_num = (d - target_dates[0]).days + 1
    date_str = d.strftime('%b %d')
    label = 'D' + str(day_num) + ' ' + date_str
    ax.annotate(label, (noon, d_high), textcoords="offset points",
                xytext=(0, 6), fontsize=8, fontweight='bold', color=c, ha='center', va='bottom')

# Axes
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.5f}'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(pd.Timestamp('2020-03-13'), pd.Timestamp('2020-03-17'))

ax.set_title('FET/USDT — Waves 1 & 2 on 15m (D=8, dev=30%)\n'
             + 'ATL = $' + f'{atl_low:.5f}' + ' | W1 = $0.01403 | W2 = $0.00892',
             fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
out = '/data/trading28/charts/fet_13_16_15m.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: ' + out)
print('Zigzag pivots: ' + str(len(zz)))
for i, p, t in zz:
    dt_str = df['dt'].iloc[i].strftime('%m-%d %H:%M')
    print('  ' + dt_str + '  ' + t + ': ' + f'{p:.5f}')
