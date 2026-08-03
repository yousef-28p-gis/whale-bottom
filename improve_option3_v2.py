"""Fixed: use individual coin config from final_bot_config.json"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

with open('final_bot_config.json') as f: configs = {r['sym']: r for r in json.load(f)}
all_coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                     if f.endswith('.json') and f!='_manifest.json'])

# Test 5 enhancements
enh_names = ['E0_Base', 'E1_+1hTrend', 'E2_+1hTrend+TightEntry', 'E3_+ConsecutiveWins', 'E4_+VolFilter']
results = {k: {'t':0,'w':0,'l':0,'pnl':0,'coins':0,'dd_sum':0} for k in enh_names}

for sym in all_coins:
    if sym not in configs: continue
    cfg = configs[sym]
    d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    
    LB, ssl_p = cfg['LB'], cfg['ssl']
    tp, sl = cfg['tp'], cfg['sl']
    
    # Whale
    sm = 3
    ln = pd.Series(l_).rolling(LB).min().values
    lc = np.zeros(n)
    for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
    sc = pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    strength = np.where(l_<=ln, (sc+hc*2)/3, 0)
    wp = pd.Series(strength).ewm(span=sm,adjust=False).mean().values
    wp_up = wp > np.roll(wp,1)
    sup = pd.Series(h).rolling(ssl_p).mean().values
    
    # Trends
    t4 = pd.Series(c).ewm(span=50*16,adjust=False).mean().values > pd.Series(c).ewm(span=200*16,adjust=False).mean().values
    t1 = pd.Series(c).ewm(span=20*4,adjust=False).mean().values > pd.Series(c).ewm(span=50*4,adjust=False).mean().values
    
    # ATR for vol filter
    atr = pd.Series(h-l_).ewm(span=14,adjust=False).mean().values
    
    # Base entries: whale up + close > SSL + close > open + 4h trend
    base_e = [i for i in range(500,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and t4[i]]
    if len(base_e) < 3: continue
    
    # E1: +1h trend
    e1 = [i for i in base_e if t1[i]]
    
    # E2: +1h trend + SSL was just crossed (tight entry)
    e2 = [i for i in e1 if c[i-1] <= sup[i-1]]  # fresh SSL cross
    
    # E4: +vol filter (above avg volume proxy)
    avg_range = np.mean(h-l_)
    e4 = [i for i in e1 if (h[i]-l_[i]) > avg_range * 1.2]
    
    # Backtest all enhancements
    for ename, entries, cooldown in [
        ('E0_Base', base_e, 12),
        ('E1_+1hTrend', e1, 12),
        ('E2_+1hTrend+TightEntry', e2, 12),
        ('E4_+VolFilter', e4, 12),
    ]:
        if len(entries) < 3: continue
        eq = [1000]; pos=0; ep=0; cool=0
        t=0; w=0; l=0; pnl=0.0
        entry_set = set(entries)
        for i in range(500, n):
            if pos:
                if h[i] >= ep*(1+tp/100):
                    pnl += tp-COMM*100; w+=1; t+=1; pos=0; cool=cooldown
                elif l_[i] <= ep*(1-sl/100):
                    pnl += -sl-COMM*100; l+=1; t+=1; pos=0; cool=cooldown
            if not pos and cool==0 and i in entry_set:
                pos=1; ep=c[i]
            if not pos and cool>0: cool-=1
            if not pos: eq.append(eq[-1])
        if pos:
            final = (c[-1]/ep-1)*100-COMM*100; pnl+=final; t+=1
            if final>0: w+=1
            else: l+=1
            eq.append(eq[-1]*(1+final/100))
        dd = ((pd.Series(eq)-pd.Series(eq).expanding().max())/pd.Series(eq).expanding().max()*100).min()
        results[ename]['t']+=t; results[ename]['w']+=w; results[ename]['l']+=l
        results[ename]['pnl']+=pnl; results[ename]['coins']+=1; results[ename]['dd_sum']+=dd

# E3: consecutive wins — dynamic cooldown (longer after loss)
for sym in all_coins:
    if sym not in configs: continue
    cfg = configs[sym]; d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    LB,ssl_p = cfg['LB'],cfg['ssl']; tp,sl = cfg['tp'],cfg['sl']
    sm=3
    ln=pd.Series(l_).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    strength=np.where(l_<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(strength).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sup=pd.Series(h).rolling(ssl_p).mean().values
    t4=pd.Series(c).ewm(span=800,adjust=False).mean().values>pd.Series(c).ewm(span=3200,adjust=False).mean().values
    entries=[i for i in range(500,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and t4[i]]
    if len(entries)<3: continue
    entry_set=set(entries)
    eq=[1000]; pos=0; ep=0; cool=0; consec_loss=0
    t=0; w=0; l=0; pnl=0.0
    for i in range(500,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl+=tp-COMM*100; w+=1; t+=1; pos=0; consec_loss=0; cool=12
            elif l_[i]<=ep*(1-sl/100):
                pnl+=-sl-COMM*100; l+=1; t+=1; pos=0; consec_loss+=1; cool=12+consec_loss*6
        if not pos and cool==0 and i in entry_set:
            pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        if not pos: eq.append(eq[-1])
    if pos:
        final=(c[-1]/ep-1)*100-COMM*100; pnl+=final; t+=1
        if final>0: w+=1
        else: l+=1
    dd=((pd.Series(eq)-pd.Series(eq).expanding().max())/pd.Series(eq).expanding().max()*100).min()
    results['E3_+ConsecutiveWins']['t']+=t; results['E3_+ConsecutiveWins']['w']+=w
    results['E3_+ConsecutiveWins']['l']+=l; results['E3_+ConsecutiveWins']['pnl']+=pnl
    results['E3_+ConsecutiveWins']['coins']+=1; results['E3_+ConsecutiveWins']['dd_sum']+=dd

print(f"{'Enhancement':<25} {'T':>5} {'WR':>7} {'W':>3} {'L':>3} {'PnL$':>9} {'$/T':>7} {'DD%':>6} {'C':>4}")
print("-"*75)
for ename in enh_names:
    r = results[ename]
    if r['t']==0: continue
    wr = r['w']/r['t']*100; avg = r['pnl']/r['t']
    dd = r['dd_sum']/r['coins'] if r['coins']>0 else 0
    print(f"{ename:<25} {r['t']:>5} {wr:>6.1f}% {r['w']:>3} {r['l']:>3} ${r['pnl']:>+8.1f} ${avg:>+6.2f} {dd:>5.1f}% {r['coins']:>4}")
