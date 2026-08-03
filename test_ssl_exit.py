"""Test SSL-based exit vs fixed SL"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])[:40]

# 4 exit methods
exit_methods = {
    'SSL_reverse': 'exit when close crosses below SSL (SMA high)',
    'Trail1%': 'trail 1% from highest high since entry',
    'Trail1.5%': 'trail 1.5% from highest high',
    'Fixed_TP1SL3': 'fixed TP1%/SL3%',
}

results = {k: {'trades':0,'wins':0,'losses':0,'pnl':0,'coins':0} for k in exit_methods}

for sym in coins:
    d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    
    # Whale indicator
    LB,sm = 50, 3
    ln = pd.Series(l_).rolling(LB).min().values
    lc = np.zeros(n)
    for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
    sc = pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    strength = np.where(l_<=ln, (sc+hc*2)/3, 0)
    wp = pd.Series(strength).ewm(span=sm,adjust=False).mean().values
    wp_up = wp > np.roll(wp,1)
    
    # SSL
    ssl_p = 10
    sup = pd.Series(h).rolling(ssl_p).mean().values
    
    # Trend 4h
    t4 = pd.Series(c).ewm(span=800,adjust=False).mean().values > pd.Series(c).ewm(span=3200,adjust=False).mean().values
    
    # Entries: whale up + close > SSL + close > open + 4h trend up
    entries = [i for i in range(300,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and t4[i]]
    if len(entries) < 3: continue
    
    has_trades = False
    for ei in entries:
        ep = c[ei]
        
        # SSL exit: exit when close < sup
        ssl_exit_pnl = None
        for j in range(ei+1, min(ei+96, n)):
            if c[j] < sup[j]:
                ssl_exit_pnl = (c[j]/ep - 1)*100 - COMM*100
                break
        if ssl_exit_pnl is None:
            ssl_exit_pnl = (c[min(ei+96,n-1)]/ep - 1)*100 - COMM*100
        
        # Trail 1%: trail 1% from peak
        trail_pnl_1 = None; peak = ep
        for j in range(ei+1, min(ei+96, n)):
            if h[j] > peak: peak = h[j]
            # Exit when price drops 1% from peak
            if l_[j] <= peak * 0.99:
                trail_pnl_1 = (peak*0.99/ep - 1)*100 - COMM*100
                break
        if trail_pnl_1 is None:
            trail_pnl_1 = (c[min(ei+96,n-1)]/ep - 1)*100 - COMM*100
        
        # Trail 1.5%
        trail_pnl_15 = None; peak15 = ep
        for j in range(ei+1, min(ei+96, n)):
            if h[j] > peak15: peak15 = h[j]
            if l_[j] <= peak15 * 0.985:
                trail_pnl_15 = (peak15*0.985/ep - 1)*100 - COMM*100
                break
        if trail_pnl_15 is None:
            trail_pnl_15 = (c[min(ei+96,n-1)]/ep - 1)*100 - COMM*100
        
        # Fixed TP1/SL3
        fix_pnl = None
        for j in range(ei+1, min(ei+96, n)):
            tp_hit = h[j] >= ep*1.01
            sl_hit = l_[j] <= ep*0.97
            if tp_hit and not sl_hit:
                fix_pnl = 1.0 - COMM*100; break
            elif sl_hit and not tp_hit:
                fix_pnl = -3.0 - COMM*100; break
            elif tp_hit and sl_hit:
                fix_pnl = 1.0 - COMM*100; break  # assume TP hit first
        if fix_pnl is None:
            fix_pnl = (c[min(ei+96,n-1)]/ep - 1)*100 - COMM*100
        
        pnls = {'SSL_reverse': ssl_exit_pnl, 'Trail1%': trail_pnl_1,
                'Trail1.5%': trail_pnl_15, 'Fixed_TP1SL3': fix_pnl}
        
        for k, pnl in pnls.items():
            results[k]['trades'] += 1
            results[k]['pnl'] += pnl
            if pnl > 0: results[k]['wins'] += 1
            else: results[k]['losses'] += 1
        has_trades = True
    
    if has_trades:
        for k in results: results[k]['coins'] += 1

print(f"{'Exit Method':<16} {'Trades':>6} {'WR':>7} {'W':>4} {'L':>4} {'PnL$':>9} {'Avg$/T':>8}")
print("-"*60)
for k in ['SSL_reverse','Trail1%','Trail1.5%','Fixed_TP1SL3']:
    r = results[k]
    if r['trades'] == 0: continue
    wr = r['wins']/r['trades']*100
    avg = r['pnl']/r['trades']
    print(f"{k:<16} {r['trades']:>6} {wr:>6.1f}% {r['wins']:>4} {r['losses']:>4} ${r['pnl']:>+8.1f} ${avg:>+7.2f}")
