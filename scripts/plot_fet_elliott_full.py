"""FET Weekly — Elliott Wave: 5-Wave Impulse + ABC (Zigzag) + B as Flat (3-3-5)"""
import json, sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'

with open('/data/trading28/data_fet_weekly.json') as f:
    candles = json.load(f)
df = pd.DataFrame(candles, columns=['ts','open','high','low','close','vol'])
df['dt'] = pd.to_datetime(df['ts'], unit='ms')

fig, ax = plt.subplots(figsize=(30, 14))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Candles
for i in range(len(df)):
    c = '#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350'
    bb = min(df['open'].iloc[i], df['close'].iloc[i])
    bh = max(abs(df['close'].iloc[i]-df['open'].iloc[i]), 0.0003)
    ax.add_patch(plt.Rectangle((mdates.date2num(df['dt'].iloc[i])-2.5, bb), 5, bh,
                               facecolor=c, edgecolor=c, alpha=0.85, linewidth=0.3))

# ===== IMPULSE (5 waves up) =====
impulse = [
    ('2020-03-09', 0.0074, '0', '#333'),
    ('2020-08-17', 0.1900, '1', '#2196F3'),
    ('2020-11-02', 0.0368, '2', '#333'),
    ('2021-03-29', 0.8787, '3', '#2196F3'),
    ('2021-06-21', 0.1587, '4', '#333'),
    ('2021-09-06', 1.1985, '5', '#2196F3'),
]

# Draw impulse lines
for i in range(len(impulse)-1):
    d1, p1, l1, c1 = impulse[i]
    d2, p2, l2, c2 = impulse[i+1]
    ax.plot([pd.Timestamp(d1), pd.Timestamp(d2)], [p1, p2], color=c1, linewidth=2.2, zorder=2)

# ===== CORRECTION ABC =====

# Wave A: Zigzag 5-3-5 (labeled a1..a5)
wave_a = [
    ('2021-09-20', 0.5800, 'a1', '#FF5722'),
    ('2021-11-08', 1.0275, 'a2', '#E91E63'),
    ('2022-02-21', 0.2255, 'a3', '#FF5722'),
    ('2022-04-04', 0.5400, 'a4', '#E91E63'),
    ('2022-07-11', 0.0662, 'a5', '#FF5722'),  # second touch of 0.0662 (double bottom)
]

# Connect W5 to A (a1)
ax.plot([pd.Timestamp('2021-09-06'), pd.Timestamp('2021-09-20')],
        [1.1985, 0.5800], color='#FF5722', linewidth=2, zorder=2)
# Wave A internal
for i in range(len(wave_a)-1):
    d1, p1, l1, c1 = wave_a[i]
    d2, p2, l2, c2 = wave_a[i+1]
    ax.plot([pd.Timestamp(d1), pd.Timestamp(d2)], [p1, p2], color=c1, linewidth=2, zorder=2)

# Wave B: Flat 3-3-5 (labeled b-a, b-b, b-c)
wave_b = [
    ('2022-08-15', 0.1179, 'b-a', '#4CAF50'),
    ('2022-10-10', 0.0731, 'b-b', '#009688'),
    ('2022-10-31', 0.1017, 'b-c', '#4CAF50'),
]

# Connect a5 to b-a
ax.plot([pd.Timestamp('2022-07-11'), pd.Timestamp('2022-08-15')],
        [0.0662, 0.1179], color='#4CAF50', linewidth=2, zorder=2)
for i in range(len(wave_b)-1):
    d1, p1, l1, c1 = wave_b[i]
    d2, p2, l2, c2 = wave_b[i+1]
    ax.plot([pd.Timestamp(d1), pd.Timestamp(d2)], [p1, p2], color=c1, linewidth=2, zorder=2)

# Wave C: Zigzag 5-3-5 (labeled c1..c5)
wave_c = [
    ('2022-11-21', 0.0527, 'c1', '#9C27B0'),
]

# Connect b-c to c1
ax.plot([pd.Timestamp('2022-10-31'), pd.Timestamp('2022-11-21')],
        [0.1017, 0.0527], color='#9C27B0', linewidth=2, zorder=2)

# ===== LABELS =====
# Impulse labels
for d, p, label, color in impulse:
    dt = pd.Timestamp(d)
    va = 'top' if label in ['1','3','5'] else 'bottom'
    yoff = -14 if va == 'top' else 14
    ax.annotate(label, (dt, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=12, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.95),
                zorder=10)

# Wave A labels
for d, p, label, color in wave_a:
    dt = pd.Timestamp(d)
    va = 'top' if label in ['a2','a4'] else 'bottom'
    yoff = -12 if va == 'top' else 12
    ax.annotate(label, (dt, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=10, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9),
                zorder=10)

# Wave B labels
for d, p, label, color in wave_b:
    dt = pd.Timestamp(d)
    va = 'top' if label in ['b-a','b-c'] else 'bottom'
    yoff = -12 if va == 'top' else 12
    ax.annotate(label, (dt, p), textcoords="offset points", xytext=(0, yoff),
                fontsize=10, fontweight='bold', color=color, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9),
                zorder=10)

# Wave C label
dt = pd.Timestamp('2022-11-21')
ax.annotate('c1', (dt, 0.0527), textcoords="offset points", xytext=(0, 14),
            fontsize=10, fontweight='bold', color='#9C27B0', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#9C27B0', alpha=0.9),
            zorder=10)

# ===== FIBONACCI ANNOTATIONS =====
fib_style = dict(fontsize=8, color='#555', ha='center', style='italic', alpha=0.85)

fibs = [
    # Impulse
    ('2020-10-01', 0.12, 'W2=0.5xW1'),
    ('2021-02-01', 0.55, 'W3=1.41xW1'),
    ('2021-05-01', 0.35, 'W4=0.5xW3'),
    ('2021-08-01', 0.72, 'W5=1.0xW3'),
    # Wave A
    ('2021-10-15', 0.92, 'a2=0.78xa1'),
    ('2022-01-01', 0.52, 'a3=2.27xa1'),
    ('2022-03-15', 0.42, 'a4=0.5xa3'),
    ('2022-05-15', 0.30, 'a5=0.786xa1'),
    # Wave B (flat)
    ('2022-09-01', 0.09, 'b-b=86.7% of b-a'),
    ('2022-10-15', 0.11, 'b-c trunc.'),
]

for d, p, txt in fibs:
    ax.annotate(txt, (pd.Timestamp(d), p), **fib_style)

# ===== SECTION LABELS (above candles) =====
section_style = dict(fontsize=13, fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', edgecolor='#aaa', alpha=0.88, pad=6))

ax.text(pd.Timestamp('2020-10-01'), 4.5, 'Impulse 5-Wave', color='#2196F3', **section_style)
ax.text(pd.Timestamp('2022-03-01'), 4.5, 'Correction ABC', color='#FF5722', **section_style)

# B wave flat label
ax.text(pd.Timestamp('2022-09-15'), 0.18, 'B = Flat\n(3-3-5)', fontsize=10,
        fontweight='bold', color='#4CAF50', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#4CAF50', alpha=0.9, pad=4))

# ===== HORIZONTAL LINES =====
for price, color, ls in [(1.1985, '#e53935', '--'), (0.0074, '#43a047', '--'),
                           (0.0662, '#FF5722', ':'), (0.0527, '#9C27B0', '-.')]:
    ax.axhline(y=price, color=color, linestyle=ls, linewidth=0.8, alpha=0.4)

# ===== AXES =====
ax.set_yscale('log')
ax.set_ylim(0.005, 5.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.4f}' if x < 0.01 else (f'${x:.3f}' if x < 1 else f'${x:.2f}')))
ax.set_xlim(df['dt'].min(), df['dt'].max())
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_title('FET/USDT — Elliott Wave: 5-Wave Impulse + ABC Correction\n'
             'A = Zigzag (5-3-5) | B = Flat (3-3-5) | C = Zigzag (5-3-5) | Weekly | Log Scale',
             fontsize=16, fontweight='bold', pad=15)

plt.tight_layout()
out = '/data/trading28/charts/fet_elliott_full.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
