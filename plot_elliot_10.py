#!/usr/bin/env python3
"""Average w5 + 10 chart plots"""
import json, numpy as np, pandas as pd, os, sys
sys.path.insert(0,'/data/trading28')
from strategies.zigzag import zigzag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD'}

with open('/data/trading28/config/shariah_coins.json') as f: sh=json.load(f)
COINS=[c for c in sh['halal']+sh['halal2'] if c not in STABLES]

DEPTH=10; DEV=1.0; D=DEPTH//2

def near(v,target,tol=0.03): return abs(v-target)<=tol

w5_sizes = []
all_patterns = []

for cn in COINS:
    fp=f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    close=df['close'].values; high=df['high'].values; low=df['low'].values
    
    pv=zigzag(high,low,DEPTH,DEV)
    if len(pv)<6: continue
    
    for i in range(len(pv)-5):
        pts=pv[i:i+6]
        if [pt[2] for pt in pts]!=['H','L','H','L','H','L']: continue
        H1,L1,H2,L2,H3,L3=pts
        w1=H1[1]-L1[1];w2=H2[1]-L1[1];w3=H2[1]-L2[1];w4=H3[1]-L2[1];w5=H3[1]-L3[1]
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3[1]>=L1[1] or L3[1]>=L2[1]: continue
        if not near(w5/(w1+w3), 0.382): continue
        w5_pct = w5/L3[1]*100
        w5_sizes.append(w5_pct)
        all_patterns.append((cn, H3, L3, w5_pct, L3[0], H3[0], df))

avg_w5 = np.mean(w5_sizes); med_w5 = np.median(w5_sizes)

print(f'عدد النماذج: {len(w5_sizes)}')
print(f'متوسط طول الموجة 5: {avg_w5:.3f}%')
print(f'وسيط: {med_w5:.3f}%')

# ── Plot 10 random patterns ──
np.random.seed(42)
indices = np.random.choice(len(all_patterns), min(10, len(all_patterns)), replace=False)

OUTC='/data/trading28/elliot_charts'
os.makedirs(OUTC, exist_ok=True)

for plot_idx, pat_idx in enumerate(indices):
    cn, H3, L3, w5_pct, i_l3, i_h3, df = all_patterns[pat_idx]
    
    # Get 200 bars around the pattern
    start = max(0, i_h3 - 80)
    end = min(len(df), i_l3 + 80)
    segment = df.iloc[start:end].copy()
    
    if len(segment) < 20: continue
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    x = range(len(segment))
    ax.plot(x, segment['close'].values, color='#00bcd4', linewidth=1, alpha=0.7, label='Close')
    
    # Shade the 5-wave area
    idx_h1 = i_h3 - H3[0] + L3[0]  # approximate
    # Just plot zigzag lines on the segment
    seg_high = segment['high'].values; seg_low = segment['low'].values
    pv_seg = zigzag(seg_high, seg_low, DEPTH, DEV)
    
    for j, pt in enumerate(pv_seg[:-1]):
        nxt = pv_seg[j+1]
        c = '#ff6b6b' if pt[2]=='H' else '#4ecdc4'
        ax.plot([pt[0], nxt[0]], [pt[1], nxt[1]], color=c, linewidth=1.5)

    # Highlight the 5-wave pattern points
    for pt in [H3, L3]:
        rel_idx = pt[0] - start
        if 0 <= rel_idx < len(segment):
            c = 'red' if pt[2]=='H' else 'lime'
            m = 'v' if pt[2]=='H' else '^'
            ax.scatter(rel_idx, pt[1], color=c, s=120, marker=m, zorder=5, edgecolors='white')
    
    # Mark L3 and H3
    rel_h3 = H3[0] - start
    rel_l3 = L3[0] - start
    if 0 <= rel_h3 < len(segment):
        ax.annotate(f'H3\n${H3[1]:.4f}', (rel_h3, H3[1]), 
                     textcoords="offset points", xytext=(0,12), ha='center', color='red', fontsize=9)
    if 0 <= rel_l3 < len(segment):
        ax.annotate(f'L3\n${L3[1]:.4f}', (rel_l3, L3[1]), 
                     textcoords="offset points", xytext=(0,-16), ha='center', color='lime', fontsize=9)
    
    # Fib levels from wave 5
    for fib_lbl, fib_pct in [('0.5', 0.5), ('1.0', 1.0)]:
        fib_price = L3[1] + fib_pct * (H3[1] - L3[1])
        ax.axhline(y=fib_price, color='gold', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(len(segment)-1, fib_price, f'Fib {fib_lbl}', color='gold', fontsize=7, va='bottom', ha='right')
    
    ax.set_title(f'{cn} — Elliot 5-Wave | w5 = {w5_pct:.2f}%', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='gray')
    ax.grid(alpha=0.15)
    ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fname = f'{OUTC}/elliot_{plot_idx+1}_{cn}.png'
    fig.savefig(fname, dpi=120, facecolor='#1a1a2e')
    plt.close(fig)
    print(f'  ✅ {fname}')

print(f'\n📊 متوسط الموجة 5: {avg_w5:.2f}% | وسيط: {med_w5:.2f}% | عدد النماذج: {len(all_patterns)}')
