#!/usr/bin/env python3
"""Test 4 classic trading strategies from literature on crypto 15m"""
import json, os, numpy as np, pandas as pd, gc
import warnings; warnings.filterwarnings('ignore')
COMM=0.002; CAP=1000; MAX_SLIPPAGE=1.5; COOLDOWN=48

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

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def rsi(c,period):
    diff=np.diff(c)
    gain=np.where(diff>0,diff,0)
    loss=np.where(diff<0,-diff,0)
    avg_gain=pd.Series(gain).ewm(span=period,adjust=False).mean().values
    avg_loss=pd.Series(loss).ewm(span=period,adjust=False).mean().values
    rs=avg_gain/(avg_loss+1e-10)
    r=np.zeros(len(c))
    r[1:]=100-(100/(1+rs))
    return r

def sim(entries,c,h,l,n,tp,sl,long_only=False):
    """Simulate trades from boolean entry array"""
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw,-sl*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0 and i in entries:
            pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: 
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

def get_entries(idx,entry_bool):
    return set(np.where(entry_bool)[0])

# ═══════════════════════════════════════════════════
# STRATEGY 1: Connors RSI 2
# ═══════════════════════════════════════════════════
def s1_connors_rsi2(c,h,l,o,ts):
    n=len(c)
    ema200=ema(c,200)
    r=rsi(c,2)
    ema5=ema(c,5)
    
    entries=set()
    for i in range(200,n):
        if c[i]>ema200[i] and r[i]<10:
            entries.add(i)
    return entries

# ═══════════════════════════════════════════════════
# STRATEGY 2: Connors RSI 25/75
# ═══════════════════════════════════════════════════
def s2_connors_rsi25(c,h,l,o,ts):
    n=len(c)
    ema200=ema(c,200)
    r=rsi(c,4)
    ema5=ema(c,5)
    
    entries=set()
    for i in range(200,n):
        if c[i]>ema200[i] and r[i]<25:
            entries.add(i)
    return entries

# ═══════════════════════════════════════════════════
# STRATEGY 3: Bollinger Band Mean Reversion
# ═══════════════════════════════════════════════════
def s3_bb_mr(c,h,l,o,ts):
    n=len(c)
    sma20=pd.Series(c).rolling(20).mean().values
    std20=pd.Series(c).rolling(20).std().values
    lower=sma20-2*std20
    
    entries=set()
    for i in range(200,n):
        if l[i]<=lower[i] and c[i-1]>lower[i-1]:  # Touch lower band
            entries.add(i)
    return entries

# ═══════════════════════════════════════════════════
# STRATEGY 4: Donchian 20 Breakout (Turtle)
# ═══════════════════════════════════════════════════
def s4_donchian20(c,h,l,o,ts):
    n=len(c)
    high20=pd.Series(h).rolling(20).max().shift(1).values
    # Break above 20-period high
    entries=set()
    for i in range(21,n):
        if h[i]>high20[i] and c[i-1]<=high20[i]:
            entries.add(i)
    return entries

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
strategies=[
    ('1.Connors_RSI2', s1_connors_rsi2, [(3,1.5),(4,2)]),
    ('2.Connors_RSI25', s2_connors_rsi25, [(3,1.5),(4,2)]),
    ('3.BB_MeanRev', s3_bb_mr, [(3,1.5),(4,2)]),
    ('4.Donchian20', s4_donchian20, [(3,1.5),(4,2),(5,2.5)]),
]

for sname, sfunc, tp_sl_grid in strategies:
    print(f"\n{'='*70}")
    print(f"{sname}")
    print(f"{'='*70}")
    
    for tp,sl in tp_sl_grid:
        prev_res=[]; cur_res=[]
        
        for sym in LIQ:
            pd_=load(DP,sym); cd_=load(DC,sym)
            if pd_ is None or cd_ is None: continue
            
            # PREV
            pc,ph,pl,po,pts=pd_
            entries=sfunc(pc,ph,pl,po,pts)
            r=sim(entries,pc,ph,pl,len(pc),tp,sl)
            if r: r['sym']=sym; prev_res.append(r)
            
            # CUR
            cc,ch,cl,co,cts=cd_
            entries=sfunc(cc,ch,cl,co,cts)
            r=sim(entries,cc,ch,cl,len(cc),tp,sl)
            if r: r['sym']=sym; cur_res.append(r)
        
        for label,res in [('PREV',prev_res),('CUR',cur_res)]:
            if not res: continue
            tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
            tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
            wr=tw/tt*100 if tt>0 else 0
            dd=np.mean([x['dd'] for x in res])
            gr=sum(1 for x in res if x['pnl']>0)
            print(f"  TP{tp}/SL{sl} {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")
        gc.collect()

print("\nDone")
