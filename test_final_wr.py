"""Final push: optimize TP/SL + filters for high WR AND profit"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])[:60]

# TP/SL combos that favor high WR
TP_SL = [
    (0.5, 3.0), (0.75, 3.0), (1.0, 3.0), (1.5, 3.0), (2.0, 3.0),
    (1.0, 2.5), (1.5, 2.5), (2.0, 2.5),
    (0.5, 2.0), (1.0, 2.0),
]

# Filters
filters = {
    'NoFilter': None,
    '4h': None,
    '1h': None,
    '4h+1h': None,
}

results = {}

for fname in filters:
    for tp, sl in TP_SL:
        key = f'{fname}_TP{tp}_SL{sl}'
        results[key] = {'t':0,'w':0,'l':0,'pnl':0,'coins':0}

for sym in coins:
    d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    
    # Whale
    LB,sm = 50, 3
    ln = pd.Series(l_).rolling(LB).min().values
    lc = np.zeros(n)
    for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
    sc = pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    strength = np.where(l_<=ln, (sc+hc*2)/3, 0)
    wp = pd.Series(strength).ewm(span=sm,adjust=False).mean().values
    wp_up = wp > np.roll(wp,1)
    sup = pd.Series(h).rolling(10).mean().values
    
    # Trends
    t4 = pd.Series(c).ewm(span=50*16,adjust=False).mean().values > pd.Series(c).ewm(span=200*16,adjust=False).mean().values
    t1 = pd.Series(c).ewm(span=20*4,adjust=False).mean().values > pd.Series(c).ewm(span=50*4,adjust=False).mean().values
    
    filters['NoFilter'] = np.ones(n, bool)
    filters['4h'] = t4
    filters['1h'] = t1
    filters['4h+1h'] = t4 & t1
    
    # Precompute entries for each filter
    entries_by_filter = {}
    for fname, filt in filters.items():
        entries_by_filter[fname] = [i for i in range(300,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and filt[i]]
    
    for fname, filt in filters.items():
        entries = entries_by_filter[fname]
        if len(entries) < 3: continue
        
        for tp, sl in TP_SL:
            key = f'{fname}_TP{tp}_SL{sl}'
            for ei in entries:
                ep = c[ei]; end = min(ei+48, n)
                th=sh=False; tj=sj=99999
                for j in range(ei+1, end):
                    if not th and h[j] >= ep*(1+tp/100): th=True; tj=j
                    if not sh and l_[j] <= ep*(1-sl/100): sh=True; sj=j
                    if th and sh: break
                results[key]['t'] += 1
                if th and not sh:
                    results[key]['w'] += 1; results[key]['pnl'] += tp - COMM*100
                elif sh and not th:
                    results[key]['l'] += 1; results[key]['pnl'] += -sl - COMM*100
                else:
                    results[key]['pnl'] += (c[end-1]/ep-1)*100 - COMM*100
            results[key]['coins'] += 1

# Print best by WR
print(f"{'Config':<22} {'T':>5} {'WR':>7} {'W':>4} {'L':>4} {'PnL$':>9} {'$/T':>7} {'C':>4}")
print("-"*65)
sorted_results = sorted([(k,v) for k,v in results.items() if v['t']>=10], key=lambda x: -x[1]['w']/x[1]['t']*100)
for k, v in sorted_results[:30]:
    wr = v['w']/v['t']*100
    avg = v['pnl']/v['t']
    print(f"{k:<22} {v['t']:>5} {wr:>6.1f}% {v['w']:>4} {v['l']:>4} ${v['pnl']:>+8.1f} ${avg:>+6.2f} {v['coins']:>4}")

# Also show best by PnL
print(f"\n🏆 BEST BY PnL:")
best_pnl = sorted([(k,v) for k,v in results.items() if v['t']>=10], key=lambda x: -x[1]['pnl'])
for k, v in best_pnl[:10]:
    wr = v['w']/v['t']*100
    print(f"{k:<22} T={v['t']:>4} WR={wr:.1f}% PnL=${v['pnl']:>+.1f}")
