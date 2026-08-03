#!/usr/bin/env python3
"""Plot 10 Elliot 5-Wave patterns with w5 fib validation"""
import json, os, sys, random
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

DATA = '/data/trading28/data/3m_4months'

def near(v, t, tol=0.05): return abs(v-t) <= tol

def find_5waves(pv):
    pats = []
    for i in range(len(pv)-5):
        p = pv[i:i+6]
        if [pt[2] for pt in p] != ['H','L','H','L','H','L']: continue
        H1=p[0]; L1=p[1]; H2=p[2]; L2=p[3]; H3=p[4]; L3=p[5]
        w1=H1[1]-L1[1]; w2=H2[1]-L1[1]; w3=H2[1]-L2[1]; w4=H3[1]-L2[1]; w5=H3[1]-L3[1]
        if w1<=0 or w2<=0 or w3<=0 or w4<=0 or w5<=0: continue
        if w2>=w1 or w3<=min(w1,w5): continue
        if H3[1]>=L1[1] or L3[1]>=L2[1]: continue
        # w5 = 0.382(w1+w3)
        if not near(w5/(w1+w3), 0.382): continue
        pats.append((p[0],p[1],p[2],p[3],p[4],p[5],w1,w2,w3,w4,w5))
    return pats

with open('/data/trading28/config/shariah_coins.json') as f: sh = json.load(f)
COINS = [c for c in sh['halal']+sh['halal2'] if c not in {'USDT','USDC','BUSD','DAI','TUSD'}]

all_pats = []
for cn in COINS[:80]:
    fp = f'{DATA}/{cn}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw = json.load(f)
    if len(raw) < 200: continue
    h = [r['h'] for r in raw]; l = [r['l'] for r in raw]
    pv = zigzag(h, l, 10, 1.0)
    if len(pv) < 6: continue
    for pat in find_5waves(pv):
        all_pats.append((cn, pat))

print(f'{len(all_pats)} patterns')
random.seed(1); picks = random.sample(all_pats, 10)

for page in range(2):
    fig, axes = plt.subplots(5, 1, figsize=(20, 24))
    fig.patch.set_facecolor('white')
    page_picks = picks[page*5:(page+1)*5]
    
    for idx, (cn, pat) in enumerate(page_picks):
        H1, L1, H2, L2, H3, L3, w1, w2, w3, w4, w5 = pat
        fp = f'{DATA}/{cn}.json'
        with open(fp) as f: raw = json.load(f)
        df = pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        pv = zigzag(df['high'].values, df['low'].values, 10, 1.0)
        n = len(df)
        pad = 30; start = max(0, H1[0]-pad); end = min(n-1, L3[0]+pad); win = end-start
        
        ax = axes[idx]; ax.set_facecolor('white')
        
        for i in range(start, end):
            ii = i-start; row = df.iloc[i]
            clr = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            ax.plot([ii,ii], [row['low'],row['high']], color=clr, linewidth=0.3, alpha=0.5)
            ax.plot([ii-0.15,ii+0.15], [row['open'],row['close']], color=clr, linewidth=1.2, alpha=0.7)
        
        zx, zy = [], []
        for pi, pr, pt in pv:
            if start <= pi <= end: zx.append(pi-start); zy.append(pr)
        if zx: ax.plot(zx, zy, color='#1565C0', linewidth=1.5, zorder=3, alpha=0.35)
        
        waves = [H1, L1, H2, L2, H3, L3]
        colors = ['#FF6D00','#00E676','#FF6D00','#00E676','#FF6D00','#00E676']
        labels = ['(0)','1','2','3','4','5']
        for wi, (pi, pr, pt) in enumerate(waves):
            x = pi-start
            if 0 <= x < win:
                ax.scatter(x, pr, color=colors[wi], s=160, zorder=8, marker='v' if pt=='H' else '^')
                offset = 1.03 if pt=='H' else 0.97
                ax.annotate(labels[wi], (x, pr*offset), fontsize=10, fontweight='bold', ha='center', color=colors[wi])
        
        wpts = [(w[0]-start, w[1]) for w in waves]
        vw = [(x,y) for x,y in wpts if 0<=x<win]
        if len(vw) >= 2:
            xs, ys = zip(*vw)
            ax.plot(xs, ys, color='#FFD600', linewidth=2.5, zorder=4, alpha=0.65)
        
        for wi in range(5):
            x0 = waves[wi][0]-start; y0 = waves[wi][1]
            x1 = waves[wi+1][0]-start; y1 = waves[wi+1][1]
            mx = (x0+x1)//2; my = (y0+y1)//2
            if 0 <= mx < win:
                ax.text(mx, my, str(wi+1), fontsize=13, color='#FFD600', fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.65))
        
        r5 = w5/(w1+w3)
        ax.set_title(f'{cn} | w1:{w1:.4f} w3:{w3:.4f} w5:{w5:.4f} | w5/(w1+w3)={r5:.3f} | Elliot Rules OK',
            fontsize=10, fontweight='bold', color='#333')
        ax.set_ylabel('USDT', color='black')
        ax.tick_params(colors='black')
        ax.grid(True, alpha=0.12)
        step = max(1, win//8); ticks = list(range(0, win, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([pd.to_datetime(df['ts'].iloc[start+i], unit='ms').strftime('%m/%d %H:%M') for i in ticks],
            rotation=45, ha='right', fontsize=6)
        ax.spines[['top','right']].set_visible(False)
        ax.spines['bottom'].set_color('#ccc'); ax.spines['left'].set_color('#ccc')
    
    plt.suptitle(f'Elliot 5-Wave + w5=0.382(w1+w3) ({page+1}/2)',
        fontsize=14, fontweight='bold', y=0.998, color='black')
    plt.tight_layout()
    out = f'/data/trading28/scripts/elliot_w5fib_{page+1}.png'
    plt.savefig(out, dpi=120, facecolor='white')
    plt.close()
    print(f'✅ {out}')

for i, (cn, _) in enumerate(picks):
    print(f'  {i+1}. {cn}')
