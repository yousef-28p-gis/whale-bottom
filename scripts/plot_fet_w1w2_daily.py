"""FET Daily — Waves 1 & 2 with Zigzag (D=12, dev=21%) — 5 waves, wave 3 extended"""
import json, sys
sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_fet_daily_w12.json') as f:
    candles = json.load(f)

df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

mask = (df['dt'] >= '2020-02-01') & (df['dt'] <= '2020-12-31')
df = df[mask].copy().reset_index(drop=True)

highs_arr = df['high'].values
lows_arr = df['low'].values

zz = zigzag(highs_arr, lows_arr, depth=12, dev=21.0)

fig, ax = plt.subplots(figsize=(24, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Candles
for i, row in df.iterrows():
    c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
    bb = min(row['open'], row['close'])
    bh = max(abs(row['close']-row['open']), 0.0001)
    w = 0.6
    ax.add_patch(plt.Rectangle((mdates.date2num(row['dt'])-w/2, bb), w, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.3))

# ---- Main Wave 1 & 2 ----
w0_d = pd.Timestamp('2020-03-13')
w1_d = pd.Timestamp('2020-08-18')
w2_d = pd.Timestamp('2020-11-04')

ax.plot([w0_d, w1_d], [0.0074, 0.1900], color='#2196F3', linewidth=3, zorder=4, alpha=0.8)
ax.plot([w1_d, w2_d], [0.1900, 0.0368], color='#FF5722', linewidth=3, zorder=4, alpha=0.8)

for d, p, label, color in [
    (w0_d, 0.0074, '0', '#333'),
    (w1_d, 0.1900, '1', '#2196F3'),
    (w2_d, 0.0368, '2', '#333'),
]:
    va = 'top' if label == '1' else 'bottom'
    yoff = -15 if va == 'top' else 15
    ax.annotate(label, (d, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=15, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=color, alpha=0.95), zorder=10)

# ---- Zigzag lines ----
zz_points = [(df['dt'].iloc[i], p, t) for (i, p, t) in zz]
for idx in range(len(zz_points)-1):
    d1, p1, t1 = zz_points[idx]
    d2, p2, t2 = zz_points[idx+1]
    ax.plot([d1, d2], [p1, p2], color='#888', linewidth=1.0, linestyle='--', zorder=3, alpha=0.6)

# Zigzag pivot dots
for d, p, t in zz_points:
    color = '#4CAF50' if t == 'H' else '#E91E63'
    marker = 'v' if t == 'H' else '^'
    ax.scatter(d, p, s=30, c=color, marker=marker, zorder=5, edgecolors='white', linewidths=0.8)

# ---- W1 internal wave labels ----
# 10 zz pivots inside W1 (rows 41 to 199):
# idx=41 L:0.0073  idx=55 H:0.0157  idx=58 L:0.0124  idx=68 H:0.0175
# idx=75 L:0.0135  idx=89 H:0.0224  idx=99 L:0.0155
# idx=136 H:0.0330  idx=147 L:0.0251  idx=199 H:0.1900
#
# 5 main waves, wave iii extended:
# i:   L:0.0073→H:0.0157   (idx 41→55)
# ii:  H:0.0157→L:0.0124   (idx 55→58)
# iii: L:0.0124→…→H:0.0330 (idx 58→136, 5 subwaves)
# iv:  H:0.0330→L:0.0251   (idx 136→147)
# v:   L:0.0251→H:0.1900   (idx 147→199)

# Label main waves at their end pivots
main_labels_zz = [
    (55, 'i', '#2196F3'),      # H:0.0157
    (58, 'ii', '#E91E63'),      # L:0.0124
    (136, 'iii', '#2196F3'),    # H:0.0330
    (147, 'iv', '#E91E63'),     # L:0.0251
    (199, 'v', '#2196F3'),      # H:0.1900
]

for zz_idx, label, color in main_labels_zz:
    d = df['dt'].iloc[zz_idx]
    p = highs_arr[zz_idx] if label in ['i','iii','v'] else lows_arr[zz_idx]
    t = 'H' if label in ['i','iii','v'] else 'L'
    va = 'top' if t == 'H' else 'bottom'
    yoff = -16 if va == 'top' else 16
    ax.annotate(label, (d, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=11, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, alpha=0.9), zorder=10)

# Subwaves inside extended iii: idx 68(H), 75(L), 89(H), 99(L)
iii_sub_labels = [
    (68, '(i)', '#666'),    # H:0.0175
    (75, '(ii)', '#888'),   # L:0.0135
    (89, '(iii)', '#666'),  # H:0.0224
    (99, '(iv)', '#888'),   # L:0.0155
]

for zz_idx, label, color in iii_sub_labels:
    d = df['dt'].iloc[zz_idx]
    t = 'H' if '(i)' in label or '(iii)' in label else 'L'
    p = highs_arr[zz_idx] if t == 'H' else lows_arr[zz_idx]
    va = 'top' if t == 'H' else 'bottom'
    yoff = -10 if va == 'top' else 10
    ax.annotate(label, (d, p), textcoords="offset points", xytext=(15, yoff),
                fontsize=8, fontweight='bold', color=color, ha='left', va=va, zorder=10)

# Highlight wave iii zone
iii_start_d = df['dt'].iloc[58]
iii_end_d = df['dt'].iloc[136]
ax.axvspan(iii_start_d, iii_end_d, alpha=0.06, color='#2196F3', zorder=1)
ax.text(iii_start_d + (iii_end_d - iii_start_d)/2, 0.22, 'Extended Wave iii\n(5 subwaves)',
        fontsize=9, fontweight='bold', color='#2196F3', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2196F3', alpha=0.85, pad=3),
        zorder=10)

# ---- Fibonacci Annotations ----
fib_style = dict(fontsize=8.5, color='#555', ha='center', style='italic',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#ddd', alpha=0.85))

# Main subwave ratios
fib_anns = [
    # (x_date, y_price, text)
    ('2020-03-22', 0.018, 'ii = 39.8% of i\n≈ Fib 0.382'),
    ('2020-05-01', 0.040, 'iii = 2.48× i\niv = 38.3% of iii\n≈ Fib 0.382'),
    ('2020-07-15', 0.12, 'v = extended'),
]

for d, p, txt in fib_anns:
    ax.annotate(txt, (pd.Timestamp(d), p), **fib_style)

# iii internal fib ratios
iii_fibs = [
    ('2020-04-05', 0.009, '(ii)=78.4%≈0.786'),
    ('2020-05-05', 0.027, '(iv)=77.5%≈0.786'),
]
for d, p, txt in iii_fibs:
    ax.annotate(txt, (pd.Timestamp(d), p), fontsize=7.5, color='#888', ha='center', style='italic')

# W2 fib retracement levels
w1_len = 0.1900 - 0.0074
for f in [0.236, 0.382, 0.5, 0.618, 0.786]:
    price = 0.1900 - (w1_len * f)
    ax.axhline(y=price, color='#bbb', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.annotate(f'{f:.3f}', (pd.Timestamp('2020-12-20'), price),
                fontsize=7, color='#999', ha='left', va='center')

w2_ret = (0.1900 - 0.0368) / w1_len * 100
ax.annotate(f'W2 = {w2_ret:.1f}%', xy=(w2_d, 0.0368),
            xytext=(50, 25), textcoords="offset points",
            fontsize=10, color='#FF5722', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#FF5722', alpha=0.85))

# ---- Axes ----
ax.set_yscale('log')
ax.set_xlim(df['dt'].min(), pd.Timestamp('2021-01-01'))
ax.set_ylim(0.005, 0.28)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.4f}' if x < 0.01 else f'${x:.3f}'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_title('FET/USDT — Waves 1 & 2 with Extended Wave iii (D=12, dev=21%)\n'
             f'Daily | W1=${0.1900:.4f} | W2=${0.0368:.4f} ({w2_ret:.1f}% retrace) | Log Scale',
             fontsize=15, fontweight='bold', pad=12)

plt.tight_layout()
out = '/data/trading28/charts/fet_w1w2_daily.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
