"""FET Weekly — LD + Zigzag 5-3-5 + Fibonacci 0→5 only"""
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

pivots = zigzag(highs, lows, depth=6, dev=1.0)

hist_low_idx = lows.argmin()
hist_high_idx = highs.argmax()

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
zz_x = [df['dt'].iloc[b] for b,_,_ in pivots]
zz_y = [p for _,p,_ in pivots]
ax1.plot(zz_x, zz_y, color='#1565C0', linewidth=2.2, zorder=5)

dt = lambda pi: df['dt'].iloc[pivots[pi][0]]
pr = lambda pi: pivots[pi][1]

# ===== FIBONACCI: 0 (P1) → 5 (P6) =====
p0, p5 = 0.0073, 1.1985
rng = p5 - p0  # $1.1912

fib_levels = [
    (0.236, '#1565C0'),
    (0.382, '#2E7D32'),
    (0.5,   '#F57F17'),
    (0.618, '#E65100'),
    (0.786, '#C62828'),
    (0.886, '#6A1B9A'),
    (1.0,   '#333333'),
]

for f_pct, color in fib_levels:
    y = p5 - rng * f_pct
    ax1.axhline(y=y, color=color, linestyle='--', linewidth=1.8, alpha=0.7, zorder=4)
    ax1.text(df['dt'].iloc[-3], y, f'{f_pct*100:.1f}%  ${y:.4f}',
             fontsize=8, color=color, va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=color, alpha=0.8))

# Fib title
ax1.text(0.84, 0.92, 'فيبو 0→5\n$0.0073→$1.1985',
         transform=ax1.transAxes, fontsize=9, color='#333', fontweight='bold', va='top', ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#333', alpha=0.92))

# ===== ELLIOTT LABELS =====

# LD: 0-1-2-3-4-5
ld = [(1,'0','#333','L'), (2,'1','#E91E63','H'), (3,'2','#FF5722','L'),
      (4,'3','#4CAF50','H'), (5,'4','#2196F3','L'), (6,'5','#9C27B0','H')]
for pi, label, color, ptype in ld:
    idx, price, pt = pivots[pi]
    d = df['dt'].iloc[idx]
    ys = 22 if pt == 'H' else -26
    ax1.scatter(d, price, s=160, c=color, zorder=9, edgecolors='white', linewidth=2)
    ax1.annotate(label, (d, price), textcoords="offset points", xytext=(0, ys),
                ha='center', fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.93))

# ABC ZIGZAG 5-3-5
# A = 5 down: P6→P9  (P7, P8 are sub-waves 1-2-3, rest compressed)
# B = 3 up: P9→P10
# C = 5 down: P10→P15 (compressed into 1 at this TF)

# A label at P9 with sub-structure note
a_idx, a_price = pivots[9][0], pivots[9][1]
ax1.scatter(df['dt'].iloc[a_idx], a_price, s=160, c='#FF6D00', zorder=9, edgecolors='white', linewidth=2)
ax1.annotate('A', (df['dt'].iloc[a_idx], a_price), textcoords="offset points", xytext=(0, -26),
            ha='center', fontsize=12, fontweight='bold', color='#FF6D00',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF6D00', alpha=0.93))

# B label at P10
b_idx, b_price = pivots[10][0], pivots[10][1]
ax1.scatter(df['dt'].iloc[b_idx], b_price, s=160, c='#795548', zorder=9, edgecolors='white', linewidth=2)
ax1.annotate('B', (df['dt'].iloc[b_idx], b_price), textcoords="offset points", xytext=(0, 22),
            ha='center', fontsize=12, fontweight='bold', color='#795548',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#795548', alpha=0.93))

# C label at P15
c_idx, c_price = pivots[15][0], pivots[15][1]
ax1.scatter(df['dt'].iloc[c_idx], c_price, s=160, c='#546E7A', zorder=9, edgecolors='white', linewidth=2)
ax1.annotate('C', (df['dt'].iloc[c_idx], c_price), textcoords="offset points", xytext=(0, -26),
            ha='center', fontsize=12, fontweight='bold', color='#546E7A',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#546E7A', alpha=0.93))

# Sub-wave markers for A (P7, P8 = first 2 of 5)
for pi, lbl in [(7,'1-2'), (8,'2-3')]:
    idx, price, _ = pivots[pi]
    d = df['dt'].iloc[idx]
    ax1.scatter(d, price, s=50, c='#FF6D00', zorder=7, alpha=0.6, marker='s')
    ax1.annotate(lbl, (d, price), textcoords="offset points", xytext=(15, 8),
                fontsize=7, color='#FF6D00', alpha=0.8)

# Sub-wave markers for C compressed into 1 move
for pi, lbl in [(11,'i'), (12,'ii'), (13,'iii'), (14,'iv')]:
    idx, price, _ = pivots[pi]
    d = df['dt'].iloc[idx]
    ax1.scatter(d, price, s=30, c='#546E7A', zorder=7, alpha=0.5, marker='s')

# STRUCTURE LEGEND
legend = (
    '═══ الترقيم ═══\n'
    '0-5: قطري قائد 5-3-5-3-5\n\n'
    'التصحيح (زجزاج 5-3-5):\n'
    'A: 5 هابطة — P6→P9\n'
    '   (P7=داخلية1, P8=داخلية2)\n'
    'B: 3 صاعدة — P9→P10\n'
    'C: 5 هابطة — P10→P15\n'
    '   (موجة واحدة هنا\n'
    '    5 علي الفريم الصغير)\n\n'
    '═══ فيبو 0→5 ═══\n'
    f'A عند: {100-(p5-a_price)/rng*100:.1f}% = 81.7%\n'
    f'B عند: {100-(p5-b_price)/rng*100:.1f}% = 55.3%\n'
    f'C عند: {100-(p5-c_price)/rng*100:.1f}% = 96.2%'
)
ax1.text(0.02, 0.55, legend, transform=ax1.transAxes, fontsize=8, color='#333',
        fontfamily='monospace', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#888', alpha=0.92),
        zorder=11)

# ATL & ATH
ax1.scatter(df['dt'].iloc[hist_low_idx], lows[hist_low_idx], marker='v', s=300, c='#C62828', zorder=10, edgecolors='white', linewidth=3)
ax1.annotate(f'ATL ${lows[hist_low_idx]:.5f}', (df['dt'].iloc[hist_low_idx], lows[hist_low_idx]),
            textcoords="offset points", xytext=(0, -40), ha='center',
            fontsize=10, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#C62828', alpha=0.9))
ax1.scatter(df['dt'].iloc[hist_high_idx], highs[hist_high_idx], marker='^', s=300, c='#2E7D32', zorder=10, edgecolors='white', linewidth=3)
ax1.annotate(f'ATH ${highs[hist_high_idx]:.2f}', (df['dt'].iloc[hist_high_idx], highs[hist_high_idx]),
            textcoords="offset points", xytext=(0, 30), ha='center',
            fontsize=10, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2E7D32', alpha=0.9))

ax1.set_yscale('log')
ax1.set_facecolor('white')
ax1.grid(True, alpha=0.25, color='#e0e0e0')
ax1.set_title('FET/USDT Weekly — LD(0-5) + Zigzag ABC(5-3-5) | Fib: 0(P1)→5(P6)',
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
print('Done')
