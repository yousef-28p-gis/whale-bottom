#!/usr/bin/env python3
"""Per-coin strategy selection — best of 12 strategies for each coin"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: d=json.load(f)
    return (np.array(d['c'],float),np.array(d['h'],float),
            np.array(d['l'],float),np.array(d['o'],float),
            d.get('ts',[]),len(d['c']))

def trends(c,ts,n):
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'c':c},index=idx)
        c4h=df['c'].resample('4h').last().dropna().values
        e50=ema(c4h,50); e200=ema(c4h,200)
        e50a=np.zeros(n); e200a=np.zeros(n)
        for i in range(n):
            j=i//16
            if j<len(e50): e50a[i]=e50[j]; e200a[i]=e200[j]
        t4=e50a>e200a
        c1h=df['c'].resample('1h').last().dropna().values
        e20=ema(c1h,20); e50h=ema(c1h,50)
        e20a=np.zeros(n); e50a2=np.zeros(n)
        for i in range(n):
            j=i//4
            if j<len(e20): e20a[i]=e20[j]; e50a2[i]=e50h[j]
        t1=e20a>e50a2
        return t4,t1
    except: return np.ones(n,bool),np.ones(n,bool)

def sim(entries, c, h, l, n, tp, sl):
    entry_set=set(entries)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and i in entry_set: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    wr=w/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return {'t':len(t),'wr':wr,'dd':dd,'pnl':eq-CAP,'w':w,'l':len(t)-w}

# Strategy builders
def whale_entries(c,h,l,o,n,LB,sp,filt):
    sm=3
    ln=pd.Series(l).shift(1).rolling(LB).min().values
    lc=np.zeros(n)
    for i in range(1,n): lc[i]=abs(l[i]-l[i-1])/l[i]*100
    sc=pd.Series(lc).ewm(span=sm,adjust=False).mean().values
    hc=pd.Series(sc).rolling(LB).max().values
    sr=np.where(l<=ln,(sc+hc*2)/3,0)
    wp=pd.Series(sr).ewm(span=sm,adjust=False).mean().values
    wp_up=wp>np.roll(wp,1)
    sma_h=pd.Series(h).rolling(sp).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(sp,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    return [i for i in range(200,n) if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[max(0,i-2)]*2 and wp[i]>0 and filt[i]]

def ema_cross_entries(c,h,l,o,n,fast,slow,filt):
    ef=ema(c,fast); es=ema(c,slow)
    return [i for i in range(200,n) if ef[i]>es[i] and ef[i-1]<=es[i-1] and c[i]>o[i] and filt[i]]

def breakout_entries(c,h,l,o,n,lb,filt):
    return [i for i in range(200,n) if c[i]>max(h[max(0,i-lb):i]) and c[i]>o[i] and filt[i]]

def supertrend_entries(c,h,l,o,n,per,mult,filt):
    atr=pd.Series(h-l).ewm(span=per,adjust=False).mean().values
    hl2=(pd.Series(h).rolling(per).max().values+pd.Series(l).rolling(per).min().values)/2
    upper=hl2+mult*atr; lower=hl2-mult*atr
    trend_up=np.ones(n,bool)
    for i in range(1,n):
        trend_up[i]=trend_up[i-1]
        if c[i]>upper[i-1]: trend_up[i]=True
        elif c[i]<lower[i-1]: trend_up[i]=False
    return [i for i in range(200,n) if trend_up[i] and not trend_up[i-1] and filt[i]]

# Define all strategies as (name, entry_func, filter_type)
# filter_type: 0=4h, 1=4h+1h
strategies = [
    ('W30/SSL10', lambda c,h,l,o,n,f: whale_entries(c,h,l,o,n,30,10,f)),
    ('W50/SSL5',  lambda c,h,l,o,n,f: whale_entries(c,h,l,o,n,50,5,f)),
    ('W50/SSL10', lambda c,h,l,o,n,f: whale_entries(c,h,l,o,n,50,10,f)),
    ('W50/SSL20', lambda c,h,l,o,n,f: whale_entries(c,h,l,o,n,50,20,f)),
    ('W70/SSL10', lambda c,h,l,o,n,f: whale_entries(c,h,l,o,n,70,10,f)),
    ('EMA20/50',  lambda c,h,l,o,n,f: ema_cross_entries(c,h,l,o,n,20,50,f)),
    ('EMA50/200', lambda c,h,l,o,n,f: ema_cross_entries(c,h,l,o,n,50,200,f)),
    ('Break20',   lambda c,h,l,o,n,f: breakout_entries(c,h,l,o,n,20,f)),
    ('SuperT10',  lambda c,h,l,o,n,f: supertrend_entries(c,h,l,o,n,10,3,f)),
]

TP_SL = [(2,1),(3,1.5),(5,2.5),(2.5,1.5),(4,2)]

# ── Optimize on PREV ──
print('Optimizing per-coin on PREV (2024-2025)...', flush=True)
coins=sorted(set(f.replace('.json','') for f in os.listdir('/data/trading28/data/whale_15m_prev')
    if f.endswith('.json') and f!='_manifest.json'))

best_configs = []

for si, sym in enumerate(coins):
    ppath = f'/data/trading28/data/whale_15m_prev/{sym}.json'
    cpath = f'/data/trading28/data/whale_15m_1y/{sym}.json'
    
    d = load(ppath)
    if d is None: continue
    c,h,l,o,ts,n = d
    # Crash filter
    skip=False
    for i in range(1,n):
        if abs(c[i]/c[i-1]-1)*100>40: skip=True; break
    if skip: continue
    
    t4,t1 = trends(c,ts,n)
    filters = [('4h',t4),('4h+1h',t4&t1)]
    
    best_pnl = -99999; best_cfg = None
    
    for sname, sfn in strategies:
        for flabel, filt in filters:
            entries = sfn(c,h,l,o,n,filt)
            if len(entries) < 5: continue
            for tp, sl in TP_SL:
                r = sim(entries, c, h, l, n, tp, sl)
                if r and r['pnl'] > best_pnl:
                    best_pnl = r['pnl']
                    best_cfg = {
                        'sym': sym, 'strat': sname, 'filt': flabel,
                        'tp': tp, 'sl': sl,
                        'prev_t': r['t'], 'prev_wr': r['wr'], 'prev_dd': r['dd'], 'prev_pnl': r['pnl']
                    }
    
    if best_cfg is None: continue
    
    # Test on CUR
    d2 = load(cpath)
    if d2:
        c2,h2,l2,o2,ts2,n2 = d2
        t4_2,t1_2 = trends(c2,ts2,n2)
        filt_cur = t4_2 if best_cfg['filt']=='4h' else (t4_2 & t1_2)
        sfn = dict(strategies)[best_cfg['strat']]
        entries = sfn(c2,h2,l2,o2,n2,filt_cur)
        r2 = sim(entries, c2, h2, l2, n2, best_cfg['tp'], best_cfg['sl'])
        if r2:
            best_cfg['cur_t']=r2['t']; best_cfg['cur_wr']=r2['wr']
            best_cfg['cur_dd']=r2['dd']; best_cfg['cur_pnl']=r2['pnl']
        else:
            best_cfg['cur_t']=0; best_cfg['cur_wr']=0; best_cfg['cur_dd']=0; best_cfg['cur_pnl']=0
    
    best_configs.append(best_cfg)
    
    if (si+1)%20 == 0:
        print(f'  {si+1}/{len(coins)}...', flush=True)
    gc.collect()

# ── Results ──
total_prev = sum(c['prev_pnl'] for c in best_configs)
total_cur = sum(c['cur_pnl'] for c in best_configs)
both_pos = sum(1 for c in best_configs if c['prev_pnl']>0 and c['cur_pnl']>0)
cur_pos = sum(1 for c in best_configs if c['cur_pnl']>0)

# Aggregate WR
all_prev_t = sum(c['prev_t'] for c in best_configs)
all_prev_w = sum(int(c['prev_t']*c['prev_wr']/100) for c in best_configs)
all_cur_t = sum(c['cur_t'] for c in best_configs if c['cur_t']>0)
all_cur_w = sum(int(c['cur_t']*c['cur_wr']/100) for c in best_configs if c['cur_t']>0)

print(f'\n═══ PER-COIN BEST STRATEGY ═══')
print(f'Coins: {len(best_configs)}')
print(f'PREV: {all_prev_t}t WR={all_prev_w/all_prev_t*100:.1f}% ${total_prev:+.0f}')
print(f'CUR:  {all_cur_t}t WR={all_cur_w/all_cur_t*100:.1f}% ${total_cur:+.0f}')
print(f'Both+: {both_pos} | CUR+: {cur_pos}')
print(f'COMBINED: ${total_prev+total_cur:+.0f}')

# Strategy distribution
from collections import Counter
strat_dist = Counter(c['strat'] for c in best_configs)
filt_dist = Counter(c['filt'] for c in best_configs)
tp_dist = Counter((c['tp'],c['sl']) for c in best_configs)
print(f'\nStrategies: {dict(strat_dist.most_common())}')
print(f'Filters: {dict(filt_dist)}')
print(f'TP/SL: {dict(tp_dist)}')

# Top 10
top = sorted(best_configs, key=lambda x:-(x['prev_pnl']+x['cur_pnl']))[:10]
print(f'\nTop 10 (combined):')
for c in top:
    print(f"  {c['sym']:<10} {c['strat']:<10} {c['filt']:<6} TP{c['tp']}/SL{c['sl']}  PREV:{c['prev_t']}t WR{c['prev_wr']:.0f}% ${c['prev_pnl']:+.0f}  CUR:{c['cur_t']}t WR{c['cur_wr']:.0f}% ${c['cur_pnl']:+.0f}")

# Save
with open('/data/trading28/per_coin_best.json','w') as f:
    json.dump(best_configs, f, indent=2)
print(f'\nSaved: per_coin_best.json')
