"""Individual optimization for WR + PnL across all 198 coins"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])

# Parameter grid
LB_vals = [30, 50, 70]
SSL_vals = [5, 10, 20]
SMOOTH = 3
TP_SL_pairs = [(0.5,2),(1,2),(1.5,2.5),(1,3),(2,3),(0.5,1.5)]

results = []

for si, sym in enumerate(coins):
    d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    
    # 4h trend filter
    t4 = pd.Series(c).ewm(span=800,adjust=False).mean().values > pd.Series(c).ewm(span=3200,adjust=False).mean().values
    
    best_score = -9999
    best_config = None
    
    for LB in LB_vals:
        # Whale indicator
        sm = SMOOTH
        ln = pd.Series(l_).rolling(LB).min().values
        lc = np.zeros(n)
        for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
        sc = pd.Series(lc).ewm(span=sm,adjust=False).mean().values
        hc = pd.Series(sc).rolling(LB).max().values
        strength = np.where(l_<=ln, (sc+hc*2)/3, 0)
        wp = pd.Series(strength).ewm(span=sm,adjust=False).mean().values
        wp_up = wp > np.roll(wp,1)
        
        for ssl_p in SSL_vals:
            sup = pd.Series(h).rolling(ssl_p).mean().values
            
            # Base entries (before TP/SL)
            entries = [i for i in range(500,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and t4[i]]
            if len(entries) < 3: continue
            
            for tp, sl in TP_SL_pairs:
                t=0; w=0; l=0; pnl=0.0
                for ei in entries:
                    ep=c[ei]; end=min(ei+48,n)
                    th=sh=False; tj=sj=99999
                    for j in range(ei+1,end):
                        if not th and h[j]>=ep*(1+tp/100): th=True; tj=j
                        if not sh and l_[j]<=ep*(1-sl/100): sh=True; sj=j
                        if th and sh: break
                    t+=1
                    if th and not sh: w+=1; pnl+=tp-COMM*100
                    elif sh and not th: l+=1; pnl+=-sl-COMM*100
                    else: pnl+=(c[end-1]/ep-1)*100-COMM*100
                
                if t < 5: continue
                wr = w/t*100
                # Score: WR-weighted, reward profitability
                score = wr * 10 + pnl  # 1% WR = $10 equivalent
                if score > best_score:
                    best_score = score
                    best_config = {
                        'sym': sym, 'LB': LB, 'ssl': ssl_p, 'tp': tp, 'sl': sl,
                        't': t, 'wr': wr, 'w': w, 'l': l, 'pnl': pnl,
                        'avg': pnl/t
                    }
    
    if best_config:
        results.append(best_config)
    
    if (si+1) % 40 == 0:
        print(f'{si+1}/{len(coins)}...')

# Summary
print(f'\n{"="*70}')
total_t = sum(r['t'] for r in results)
total_w = sum(r['w'] for r in results)
total_l = sum(r['l'] for r in results)
total_pnl = sum(r['pnl'] for r in results)
avg_wr = total_w/total_t*100 if total_t>0 else 0

print(f'Total: {len(results)} coins | {total_t} trades | WR={avg_wr:.1f}% | PnL=${total_pnl:+.1f}')

# Count coins with WR>55%
high_wr = [r for r in results if r['wr']>=55]
print(f'\nWR>=55%: {len(high_wr)} coins')
if high_wr:
    high_pnl = sum(r['pnl'] for r in high_wr)
    high_t = sum(r['t'] for r in high_wr)
    high_w = sum(r['w'] for r in high_wr)
    high_wr_avg = high_w/high_t*100
    print(f'  Trades: {high_t} | WR: {high_wr_avg:.1f}% | PnL: ${high_pnl:+.1f}')

# Top 20 by WR
print(f'\n🏆 Top 20 by WR:')
top = sorted(results, key=lambda x: -x['wr'])[:20]
for r in top:
    print(f"{r['sym']:<12} LB{r['LB']}/SSL{r['ssl']} TP{r['tp']}/SL{r['sl']} "
          f"T={r['t']:>3} WR={r['wr']:.1f}% PnL=${r['pnl']:>+7.1f}")

# Top 20 by PnL
print(f'\n💰 Top 20 by PnL:')
top_pnl = sorted(results, key=lambda x: -x['pnl'])[:20]
for r in top_pnl:
    print(f"{r['sym']:<12} LB{r['LB']}/SSL{r['ssl']} TP{r['tp']}/SL{r['sl']} "
          f"T={r['t']:>3} WR={r['wr']:.1f}% PnL=${r['pnl']:>+7.1f}")

# Save best config
with open('best_wr_config.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to best_wr_config.json')
