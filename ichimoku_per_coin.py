#!/usr/bin/env python3
"""Per-coin optimization: Ichimoku 8h — best params per coin based on PREV, validate on CUR"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=2

DP='/data/trading28/data/whale_15m_prev'
DC='/data/trading28/data/whale_15m_1y'
LIQ=['BTC','ETH','SOL','BNB','XRP','ADA','DOGE','AVAX','DOT','LINK',
     'MATIC','UNI','LTC','ATOM','ETC','FIL','APT','ARB','OP','NEAR',
     'INJ','TIA','SUI','SEI','PEPE','WIF']

def load(d,s):
    p=os.path.join(d,f'{s}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j=json.load(f)
    return (np.array(j['c'],float),np.array(j['h'],float),np.array(j['l'],float),np.array(j['o'],float),j.get('ts',[]))

def resample_8h(c,h,l,o,ts):
    try:
        idx=pd.to_datetime(np.array(ts),unit='ms')
        df=pd.DataFrame({'o':o,'h':h,'l':l,'c':c},index=idx)
        r=df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values,r['h'].values,r['l'].values,r['o'].values
    except: return None

def ichimoku_backtest(c,h,l,o,tenkan,kijun,senkou,tp,sl,cooldown):
    n=len(c)
    if n<senkou+30: return None
    h_t=pd.Series(h).rolling(tenkan).max().values; l_t=pd.Series(l).rolling(tenkan).min().values
    t_arr=(h_t+l_t)/2
    h_k=pd.Series(h).rolling(kijun).max().values; l_k=pd.Series(l).rolling(kijun).min().values
    k_arr=(h_k+l_k)/2
    h_s=pd.Series(h).rolling(senkou).max().values; l_s=pd.Series(l).rolling(senkou).min().values
    sb_raw=(h_s+l_s)/2; sa_raw=(t_arr+k_arr)/2
    shift=kijun
    sa=np.full(n,np.nan); sb=np.full(n,np.nan)
    for i in range(shift,n-shift):
        if i+shift<n: sa[i+shift]=sa_raw[i]
    for i in range(senkou,n-shift):
        if i+shift<n: sb[i+shift]=sb_raw[i]
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(senkou+shift,n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top=max(sa[i],sb[i]); cloud_bot=min(sa[i],sb[i])
        above=c[i]>cloud_top; below=c[i]<cloud_bot
        golden=t_arr[i]>k_arr[i] and t_arr[i-1]<=k_arr[i-1]
        death=t_arr[i]<k_arr[i] and t_arr[i-1]>=k_arr[i-1]
        if pos:
            if side==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
            else:
                if l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=cooldown
        if not pos and cool==0:
            if above and golden: pos=1; ep=c[i]; side=1
            elif below and death: pos=1; ep=c[i]; side=-1
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ── Param grid ──
params = [
    (3,9,18,'Ultra 3/9/18'),
    (5,13,26,'Fast 5/13/26'),
    (7,22,44,'Crypto 7/22/44'),
    (9,26,52,'Standard 9/26/52'),
    (10,30,60,'Mid 10/30/60'),
    (13,34,68,'Slow 13/34/68'),
]
tp_sl_grid = [(3,1.5),(4,2),(5,2.5),(6,3),(8,4)]

print("=== PER-COIN ICHIMOKU 8h OPTIMIZATION ===\n")

per_coin_best = {}

for sym in LIQ:
    pd_=load(DP,sym); cd_=load(DC,sym)
    if pd_ is None or cd_ is None: continue
    
    c4,h4,l4,o4=resample_8h(pd_[0],pd_[1],pd_[2],pd_[3],pd_[4])
    c4c,h4c,l4c,o4c=resample_8h(cd_[0],cd_[1],cd_[2],cd_[3],cd_[4])
    if c4 is None or c4c is None: continue
    
    best_prev = -99999; best_cfg = None
    
    for tenkan,kijun,senkou,pname in params:
        for tp,sl in tp_sl_grid:
            r=ichimoku_backtest(c4,h4,l4,o4,tenkan,kijun,senkou,tp,sl,COOLDOWN)
            if r and r['pnl'] > best_prev and r['t']>=10:
                best_prev = r['pnl']
                best_cfg = (pname,tp,sl,r)
    
    if best_cfg is None: continue
    
    # Test best on CUR
    pname,tp,sl,prev_r = best_cfg
    tenkan,kijun,senkou = {p[3]:p[:3] for p in params}[pname]
    cur_r = ichimoku_backtest(c4c,h4c,l4c,o4c,tenkan,kijun,senkou,tp,sl,COOLDOWN)
    
    if cur_r is None: continue
    
    per_coin_best[sym] = {
        'strat': pname, 'tp': tp, 'sl': sl,
        'prev_t': prev_r['t'], 'prev_wr': prev_r['wr'], 'prev_dd': prev_r['dd'], 'prev_pnl': prev_r['pnl'],
        'cur_t': cur_r['t'], 'cur_wr': cur_r['wr'], 'cur_dd': cur_r['dd'], 'cur_pnl': cur_r['pnl'],
    }
    
    print(f"{sym}: {pname} TP{tp}/SL{sl} | PREV {prev_r['t']}T WR={prev_r['wr']:.0f}% DD={prev_r['dd']:.1f}% ${prev_r['pnl']:+.0f} | CUR {cur_r['t']}T WR={cur_r['wr']:.0f}% ${cur_r['pnl']:+.0f}")

# ── Summary ──
total_prev=sum(c['prev_pnl'] for c in per_coin_best.values())
total_cur=sum(c['cur_pnl'] for c in per_coin_best.values())
prev_coins=sum(1 for c in per_coin_best.values() if c['prev_pnl']>0 and c['cur_pnl']>0)
all_coins=len(per_coin_best)
prev_t=sum(c['prev_t'] for c in per_coin_best.values())
cur_t=sum(c['cur_t'] for c in per_coin_best.values())
prev_w=sum(round(c['prev_t']*c['prev_wr']/100) for c in per_coin_best.values())
cur_w=sum(round(c['cur_t']*c['cur_wr']/100) for c in per_coin_best.values())

print(f"\n{'='*60}")
print(f"📋 PER-COIN SUMMARY")
print(f"{'='*60}")
print(f"Coins optimized: {all_coins}")
print(f"PREV: {prev_t}T, 🟢{prev_w} 🔴{prev_t-prev_w}, WR={prev_w/prev_t*100:.1f}%, ${total_prev:+,.0f}")
print(f"CUR:  {cur_t}T, 🟢{cur_w} 🔴{cur_t-cur_w}, WR={cur_w/cur_t*100:.1f}%, ${total_cur:+,.0f}")
print(f"COMBINED: ${total_prev+total_cur:+,.0f}")
print(f"Coins profitable in BOTH periods: {prev_coins}/{all_coins}")

# Show which params won most
from collections import Counter
strat_wins=Counter(c['strat'] for c in per_coin_best.values())
print(f"\nStrategy distribution: {strat_wins.most_common()}")

# Save
with open('/data/trading28/ichimoku_8h_per_coin.json','w') as f:
    json.dump(per_coin_best,f,indent=2)
print(f"\nSaved: ichimoku_8h_per_coin.json")
print("Done")
