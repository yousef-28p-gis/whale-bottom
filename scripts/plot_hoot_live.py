#!/usr/bin/env python3
"""Plot 2 active حوت الموجات trades from state file"""
import json, pandas as pd, numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

STATE = '/data/trading28/hoot_state.json'
CACHE = '/data/trading28/cache/hoot_live'

with open(STATE) as f: state = json.load(f)
positions = state['active']

fig, axes = plt.subplots(len(positions), 1, figsize=(18, 8*len(positions)))
if len(positions) == 1: axes = [axes]
fig.patch.set_facecolor('white')

for idx, pos in enumerate(positions):
    sym = pos['symbol']
    fpath = f'{CACHE}/{sym}USDT.json'
    if not os.path.exists(fpath): continue
    
    with open(fpath) as f: raw = json.load(f)
    df = pd.DataFrame(raw)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    n = len(close)
    
    pv = zigzag(high, low, 10, 1.0)
    
    ax = axes[idx]; ax.set_facecolor('white')
    
    # Candlesticks (last 80 bars)
    start = max(0, n - 80)
    for i in range(start, n):
        ii = i - start; row = df.iloc[i]
        clr = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([ii,ii], [row['low'], row['high']], color=clr, linewidth=0.5, alpha=0.7)
        ax.plot([ii-0.2, ii+0.2], [row['open'], row['close']], color=clr, linewidth=1.8, alpha=0.9)
    
    # Zigzag
    zx, zy = [], []
    for pi, pr, pt in pv:
        if pi >= start:
            zx.append(pi - start); zy.append(pr)
    if zx:
        ax.plot(zx, zy, color='#1565C0', linewidth=2, zorder=4, alpha=0.4)
    
    # Mark last 4 pivots (if pattern)
    if len(pv) >= 4:
        for pi, pr, pt in pv[-4:]:
            if pi >= start:
                x = pi - start
                if pt == 'H':
                    ax.scatter(x, pr, color='#FF6D00', s=130, zorder=8, marker='v')
                else:
                    ax.scatter(x, pr, color='#00E676', s=130, zorder=8, marker='^')
    
    # Find the last H1-L1-H2-L2 pattern
    pats_pv = []
    for i in range(len(pv)-3):
        p0,p1,p2,p3 = pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            pats_pv.append((p0,p1,p2,p3))
    
    # Entry bar (last candle)
    entry_bar = n - 1
    entry_x = entry_bar - start
    
    ep = pos['entry_price']; sl = pos['sl']; tp = pos['tp_full']
    tp_h = pos['tp_half']; be = pos['be']
    
    if 0 <= entry_x < (n-start):
        ax.scatter(entry_x, ep, color='yellow', s=250, zorder=10, marker='o', edgecolors='white', linewidths=3)
        ax.annotate(f'ENTRY\n${ep:.5f}', (entry_x, ep),
            xytext=(entry_x, ep * 1.01), fontsize=10, color='black', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Draw SL/TP/BE lines to right edge
        x_end = n - start - 1
        ax.axhline(y=sl, xmin=entry_x/(n-start), xmax=1.0, color='red', linewidth=2, linestyle='--', alpha=0.7, label=f'SL {sl:.5f}')
        ax.axhline(y=tp, xmin=entry_x/(n-start), xmax=1.0, color='lime', linewidth=2, linestyle='--', alpha=0.7, label=f'TP 1% {tp:.5f}')
        ax.axhline(y=tp_h, xmin=entry_x/(n-start), xmax=1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.5, label=f'Half TP {tp_h:.5f}')
        ax.axhline(y=be, xmin=entry_x/(n-start), xmax=1.0, color='orange', linewidth=1.5, linestyle=':', alpha=0.5, label=f'BE {be:.5f}')
        
        # Fill trade area
        ax.fill_between([entry_x, x_end], ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.08, color='white')
        
        ax.legend(loc='upper left', fontsize=8, facecolor='white', edgecolor='#cccccc', labelcolor='black')
    
    # Title
    t_entry = pd.to_datetime(df['ts'].iloc[entry_bar], unit='ms')
    ax.set_title(f'{sym} | دخول {t_entry.strftime("%m/%d %H:%M")} | سعر {ep:.6f} | SL {sl:.6f}',
        fontsize=13, fontweight='bold', color='#006600')
    ax.set_ylabel('USDT', color='black')
    ax.tick_params(colors='black')
    ax.grid(True, alpha=0.15)
    
    # Time axis
    step = 10; ticks = list(range(0, n-start, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(df['ts'].iloc[start+i], unit='ms').strftime('%H:%M') for i in ticks],
        rotation=45, ha='right', fontsize=7)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('حوت الموجات — صفقات حية | Half TP 0.5% + BE | MAX_POS=2',
    fontsize=15, fontweight='bold', y=0.995, color='black')
plt.tight_layout()
out = '/data/trading28/scripts/hoot_live_trades.png'
plt.savefig(out, dpi=150, facecolor='white')
print(f'✅ {out}')
for pos in positions:
    print(f"  {pos['symbol']}: {pos['entry_price']:.6f} | SL {pos['sl']:.6f} | TP {pos['tp_full']:.6f}")
