#!/usr/bin/env python3
"""Book strategies — CORRECT implementation with dynamic exits"""
import json, os, numpy as np, pandas as pd
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
def rsi(c,p):
    diff=np.diff(c); gain=np.where(diff>0,diff,0); loss=np.where(diff<0,-diff,0)
    ag=pd.Series(gain).ewm(span=p,adjust=False).mean().values
    al=pd.Series(loss).ewm(span=p,adjust=False).mean().values
    rs=ag/(al+1e-10); r=np.zeros(len(c)); r[1:]=100-(100/(1+rs)); return r

# ═══════════════════════════════════════════════════
# 1. Connors RSI 2 — PROPER: long-only, EMA200 filter, exit on close>EMA5
# ═══════════════════════════════════════════════════
def s1_connors(c,h,l,o,ts):
    n=len(c); e200=ema(c,200); r=rsi(c,2); e5=ema(c,5)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+5/100):  # TP 5%
                pnl=5-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-2.5/100):  # SL 2.5%
                pnl=max((c[i]/ep-1)*100-COMM*100,-2.5*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif c[i]>e5[i] and c[i-1]<=e5[i-1]:  # Exit on close>EMA5
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if c[i]>e200[i] and r[i]<10: pos=1; ep=c[i]  # Buy signal
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# 2. Connors RSI 25/75 — proper exit at RSI>75 or close>EMA5
# ═══════════════════════════════════════════════════
def s2_connors25(c,h,l,o,ts):
    n=len(c); e200=ema(c,200); r=rsi(c,4); e5=ema(c,5)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+5/100):
                pnl=5-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-2.5/100):
                pnl=max((c[i]/ep-1)*100-COMM*100,-2.5*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif r[i]>75 or c[i]>e5[i]:  # Exit on RSI>75 or close>EMA5
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if c[i]>e200[i] and r[i]<25: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# 3. Bollinger Band MR — exit at middle band (SMA20)
# ═══════════════════════════════════════════════════
def s3_bb(c,h,l,o,ts):
    n=len(c)
    b20=pd.Series(c).rolling(20).mean().values  # middle
    std=pd.Series(c).rolling(20).std().values
    lower=b20-2*std
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+5/100):
                pnl=5-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-2.5/100):
                pnl=max((c[i]/ep-1)*100-COMM*100,-2.5*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif c[i]>=b20[i]:  # Exit at middle band
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if l[i]<=lower[i] and c[i-1]>lower[i-1]:  # Touch lower band
                pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# 4. Turtle: Donchian 20 entry, Donchian 10 exit
# ═══════════════════════════════════════════════════
def s4_turtle(c,h,l,o,ts):
    n=len(c)
    h20=pd.Series(h).rolling(20).max().shift(1).values
    l10=pd.Series(l).rolling(10).min().shift(1).values  # exit
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    atr=pd.Series(np.maximum(h-l,np.maximum(abs(h-pd.Series(c).shift(1)),abs(l-pd.Series(c).shift(1))))).rolling(14).mean().values
    
    for i in range(200,n):
        if pos:
            if l[i]<=l10[i]:  # Exit on 10-day low
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-2/100):  # Hard SL 2%
                pnl=max((c[i]/ep-1)*100-COMM*100,-2*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if h[i]>h20[i] and c[i-1]<=h20[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# 5. NEW: Mean Reversion — Buy after N consecutive red candles
# ═══════════════════════════════════════════════════
def s5_candle_mr(c,h,l,o,ts):
    n=len(c); e200=ema(c,200)
    red=(c<o).astype(int)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+3/100):
                pnl=3-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-1.5/100):
                pnl=max((c[i]/ep-1)*100-COMM*100,-1.5*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if c[i]>e200[i] and red[i-3:i].sum()>=3:  # 3 consecutive red candles above EMA200
                pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# 6. NEW: Keltner Channel Breakout (Linda Raschke)
# ═══════════════════════════════════════════════════
def s6_keltner(c,h,l,o,ts):
    n=len(c)
    e20=ema(c,20)
    atr=pd.Series(np.maximum(h-l,np.maximum(abs(h-pd.Series(c).shift(1)),abs(l-pd.Series(c).shift(1))))).rolling(10).mean().values
    upper=e20+1.5*atr; lower=e20-1.5*atr
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    for i in range(200,n):
        if pos:
            if side==1:
                if l[i]<=e20[i]:  # Exit back to middle
                    pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-2/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-2*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            else:
                if h[i]>=e20[i]:
                    pnl=(1-c[i]/ep)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+2/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-2*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        if not pos and cool==0:
            if h[i]>=upper[i] and c[i-1]<upper[i-1]: pos=1; ep=c[i]; side=1
            elif l[i]<=lower[i] and c[i-1]>lower[i-1]: pos=1; ep=c[i]; side=-1
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
    t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ═══════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════
all_strats=[
    ('1.Connors_RSI2_proper', s1_connors),
    ('2.Connors_RSI25_proper', s2_connors25),
    ('3.BB_MR_proper', s3_bb),
    ('4.Turtle_Donchian', s4_turtle),
    ('5.Candle_3Red_MR', s5_candle_mr),
    ('6.Keltner_BO', s6_keltner),
]

for sname, sfunc in all_strats:
    print(f"\n{'='*60}")
    print(f"{sname}")
    print(f"{'='*60}")
    
    prev_res=[]; cur_res=[]
    for sym in LIQ:
        pd_=load(DP,sym); cd_=load(DC,sym)
        if pd_ is None or cd_ is None: continue
        
        pc,ph,pl,po,pts=pd_
        r=sfunc(pc,ph,pl,po,pts)
        if r: r['sym']=sym; prev_res.append(r)
        
        cc,ch,cl,co,cts=cd_
        r=sfunc(cc,ch,cl,co,cts)
        if r: r['sym']=sym; cur_res.append(r)
    
    for label,res in [('PREV',prev_res),('CUR',cur_res)]:
        if not res: continue
        tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
        tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
        wr=tw/tt*100 if tt>0 else 0
        dd=np.mean([x['dd'] for x in res])
        gr=sum(1 for x in res if x['pnl']>0)
        print(f"  {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")

print("\nDone")
