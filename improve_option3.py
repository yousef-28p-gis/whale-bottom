"""Option 3 improvements — test 5 enhancements on top 40 coins"""
import json, os, numpy as np, pandas as pd
COMM, DATA = 0.002, 'data/whale_15m_1y'

def load(sym):
    p = os.path.join(DATA, f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return {'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
            'l': np.array(d['l'],float), 'o': np.array(d['o'],float)}

# Load best existing config
with open('final_bot_config.json') as f: old_configs = {r['sym']: r for r in json.load(f)}
coins = sorted([f.replace('.json','') for f in os.listdir(DATA) 
                if f.endswith('.json') and f!='_manifest.json'])[:50]

# ── Base: Option 3 as-is (verify) ──
# ── Enhancement 1: +1h trend filter ──
# ── Enhancement 2: Trailing stop after TP1 ──
# ── Enhancement 3: Entry on pullback (not breakout) ──
# ── Enhancement 4: Dynamic SL = 1.5x ATR ──
# ── Enhancement 5: 1h+4h trend + trailing stop ──

TP, SL = 5.0, 2.5  # base from option 3

results = {}
enhancements = {
    'E0_Base(Option3)': None,  # just verify
    'E1_+1hTrend': None,
    'E2_TrailStop': None,
    'E3_PullbackEntry': None,
    'E4_DynamicSL': None,
    'E5_Trail+DynSL+1h': None,
}
for ename in enhancements:
    results[ename] = {'t':0,'w':0,'l':0,'pnl':0,'coins':0,'dd_sum':0}

for sym in coins:
    d = load(sym)
    if d is None or len(d['c'])<500: continue
    c,h,l_,o = d['c'], d['h'], d['l'], d['o']; n = len(c)
    
    cfg = old_configs.get(sym, {'LB':50,'ssl':10})
    LB = cfg.get('LB', 50); ssl_p = cfg.get('ssl', 10)
    
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
    
    # ATR
    atr = pd.Series(h-l_).ewm(span=14,adjust=False).mean().values
    
    # Precompute entries for each enhancement
    base_entries = [i for i in range(500,n) if wp_up[i] and c[i]>sup[i] and c[i]>o[i] and t4[i]]
    if len(base_entries) < 3: continue
    
    # E1 entries: base + 1h trend
    e1_entries = [i for i in base_entries if t1[i]]
    
    # E3 entries: pullback (whale was up 2-4 bars ago, price dipped, now green)
    e3_entries = []
    for i in range(500, n):
        had_whale = any(wp_up[max(0,i-k)] for k in [2,3,4])
        dipped = c[i] < c[max(0,i-2)]
        if had_whale and dipped and t4[i] and c[i] > sup[i] and c[i] > o[i]:
            e3_entries.append(i)
    
    has_results = False
    for ename, entries in [
        ('E0_Base(Option3)', base_entries),
        ('E1_+1hTrend', e1_entries),
        ('E3_PullbackEntry', e3_entries),
    ]:
        if len(entries) < 3: continue
        has_results = True
        
        # Standard backtest
        eq = [1000]; pos = 0; ep = 0; cool = 0
        t=0; w=0; l=0; pnl=0.0
        for i in range(500, n):
            if pos:
                tp_hit = h[i] >= ep*(1+TP/100)
                sl_hit = l_[i] <= ep*(1-SL/100)
                if tp_hit and not sl_hit:
                    pnl += TP-COMM*100; w+=1; t+=1; pos=0; cool=12
                    eq.append(eq[-1]*(1+(TP-COMM*100)/100))
                elif sl_hit and not tp_hit:
                    pnl += -SL-COMM*100; l+=1; t+=1; pos=0; cool=12
                    eq.append(eq[-1]*(1+(-SL-COMM*100)/100))
                elif tp_hit and sl_hit:
                    pnl += TP-COMM*100; w+=1; t+=1; pos=0; cool=12
                    eq.append(eq[-1]*(1+(TP-COMM*100)/100))
            if not pos and cool==0 and i in entries:
                pos=1; ep=c[i]
            if not pos and cool>0: cool-=1
            if not pos: eq.append(eq[-1])
        if pos:
            final = (c[-1]/ep-1)*100-COMM*100; pnl+=final; t+=1
            if final>0: w+=1
            else: l+=1
        dd = ((pd.Series(eq)-pd.Series(eq).expanding().max())/pd.Series(eq).expanding().max()*100).min()
        results[ename]['t']+=t; results[ename]['w']+=w; results[ename]['l']+=l
        results[ename]['pnl']+=pnl; results[ename]['coins']+=1; results[ename]['dd_sum']+=dd
    
    # E2 & E4 & E5: trailing stop variants
    for ename, entries, use_trail, dynamic_sl in [
        ('E2_TrailStop', base_entries, True, False),
        ('E4_DynamicSL', base_entries, False, True),
        ('E5_Trail+DynSL+1h', e1_entries, True, True),
    ]:
        if len(entries) < 3: continue
        eq = [1000]; pos=0; ep=0; peak=0; cool=0
        t=0; w=0; l=0; pnl=0.0
        for i in range(500, n):
            sl_pct = atr[i]*1.5/c[i]*100 if dynamic_sl else SL
            if pos:
                if h[i] > peak: peak = h[i]
                trail_sl = peak*(1-(1 if use_trail else sl_pct)/100)
                tp_hit = h[i] >= ep*(1+TP/100)
                sl_hit = l_[i] <= trail_sl
                if tp_hit:
                    pnl += TP-COMM*100; w+=1; t+=1; pos=0; cool=12
                elif sl_hit:
                    loss = (trail_sl/ep-1)*100-COMM*100
                    pnl += loss; l+=1; t+=1; pos=0; cool=12
            if not pos and cool==0 and i in entries:
                pos=1; ep=c[i]; peak=ep
            if not pos and cool>0: cool-=1
        if pos:
            final = (c[-1]/ep-1)*100-COMM*100; pnl+=final; t+=1
            if final>0: w+=1
            else: l+=1
        results[ename]['t']+=t; results[ename]['w']+=w; results[ename]['l']+=l
        results[ename]['pnl']+=pnl; results[ename]['coins']+=1

# Print
print(f"{'Enhancement':<22} {'T':>5} {'WR':>7} {'W':>3} {'L':>3} {'PnL$':>9} {'$/T':>7} {'DD%':>6} {'C':>4}")
print("-"*70)
for ename in ['E0_Base(Option3)','E1_+1hTrend','E2_TrailStop','E3_PullbackEntry','E4_DynamicSL','E5_Trail+DynSL+1h']:
    r = results.get(ename)
    if not r or r['t']==0: continue
    wr = r['w']/r['t']*100; avg = r['pnl']/r['t']
    dd = r['dd_sum']/r['coins']
    print(f"{ename:<22} {r['t']:>5} {wr:>6.1f}% {r['w']:>3} {r['l']:>3} ${r['pnl']:>+8.1f} ${avg:>+6.2f} {dd:>5.1f}% {r['coins']:>4}")
