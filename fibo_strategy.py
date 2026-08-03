#!/usr/bin/env python3
"""Fibonacci Retracement + EMA200 Trend Strategy (from Numerical Analysis book)"""
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

def fibo_strategy(c,h,l,o,ts,tp_pct,sl_pct,fibo_level,lookback=96,trend_ema=200):
    """
    Strategy from التحليل الرقمي:
    1. Find recent swing high/low (using lookback candles)
    2. Price must retrace to fibo_level (e.g. 0.50 or 0.618) of the move
    3. EMA200 trend filter: longs above EMA, shorts below EMA
    4. Entry on first candle that touches fibo level and reverses
    5. Fixed TP/SL
    """
    n=len(c); e200=ema(c,trend_ema)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    
    # Track swing high/low with rolling window
    for i in range(lookback+50, n):
        # ── Find recent swing ──
        start = i - lookback
        recent_h = h[start:i].max()
        recent_l = l[start:i].min()
        h_idx = start + np.argmax(h[start:i])
        l_idx = start + np.argmin(l[start:i])
        
        # Determine direction: which came last? 
        # If high after low → uptrend (price went up)
        # If low after high → downtrend (price went down)
        
        if pos:
            # ── Exit management ──
            if side==1:
                if h[i]>=ep*(1+tp_pct/100):
                    pnl=tp_pct-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-sl_pct/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl_pct*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            else:
                if l[i]<=ep*(1-tp_pct/100):
                    pnl=tp_pct-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+sl_pct/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl_pct*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        
        if not pos and cool==0:
            range_h_l = recent_h - recent_l
            if range_h_l <= 0: continue
            
            if h_idx > l_idx:  # Uptrend (high was after low)
                # Look for long: price retraced to fibo level
                fibo_price = recent_h - range_h_l * fibo_level
                
                # Check: price touched fibo level and is above EMA200
                if (l[i] <= fibo_price * 1.002 and l[i] >= fibo_price * 0.998 
                    and c[i] > fibo_price  # reversal candle
                    and c[i] > e200[i]):  # trend filter
                    pos=1; ep=c[i]; side=1
            
            else:  # Downtrend (low was after high)
                fibo_price = recent_l + range_h_l * fibo_level
                
                if (h[i] >= fibo_price * 0.998 and h[i] <= fibo_price * 1.002
                    and c[i] < fibo_price  # reversal down
                    and c[i] < e200[i]):  # trend filter
                    pos=1; ep=c[i]; side=-1
        
        if not pos and cool>0: cool-=1
        cv.append(eq)
    
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

# ── Test grid ──
print("=== Fibonacci Retracement + EMA200 Strategy ===")
print("From: التحليل الرقمي - Dr. Sherif Aborehab\n")

for fibo in [0.382, 0.50, 0.618]:
    for tp,sl in [(3,1.5),(4,2),(5,2.5)]:
        print(f"\n--- Fibo={fibo*100:.0f}% | TP{tp}/SL{sl} ---")
        pr=[]; cr=[]
        for sym in LIQ:
            pd_=load(DP,sym); cd_=load(DC,sym)
            if pd_ is None or cd_ is None: continue
            
            r=fibo_strategy(*pd_,tp,sl,fibo)
            if r: r['sym']=sym; pr.append(r)
            
            r=fibo_strategy(*cd_,tp,sl,fibo)
            if r: r['sym']=sym; cr.append(r)
        
        for label,res in [('PREV',pr),('CUR',cr)]:
            if not res: continue
            tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
            tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
            wr=tw/tt*100 if tt>0 else 0
            dd=np.mean([x['dd'] for x in res]); gr=sum(1 for x in res if x['pnl']>0)
            print(f"  {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")

print("\nDone")
