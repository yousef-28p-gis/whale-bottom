#!/usr/bin/env python3
"""Role Reversal Strategy - Support becomes Resistance (and vice versa)"""
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

def role_reversal(c,h,l,o,ts,tp_pct,sl_pct,lookback=48):
    """
    Strategy from استراتيجية تبادل الأدوار by Adam Barnawi:
    1. Identify resistance breakout (price breaks above recent peak)
    2. Wait for price to retest the broken level (now support)
    3. Look for indecision candles (doji, small body)
    4. Enter long above the breakout candle
    5. Stop below the retest candle
    6. Target: previous peak or fibo level
    """
    n=len(c); e200=ema(c,200)
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; stop_price=0
    
    for i in range(lookback+20, n):
        # Find recent resistance: highest high in lookback
        recent_high = h[i-lookback:i].max()
        high_idx = i - lookback + np.argmax(h[i-lookback:i])
        
        if pos:
            # Exit management
            if h[i]>=ep*(1+tp_pct/100):
                pnl=tp_pct-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            elif l[i]<=ep*(1-sl_pct/100):
                pnl=max((c[i]/ep-1)*100-COMM*100,-sl_pct*MAX_SLIPPAGE-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        
        if not pos and cool==0:
            # Check if price recently broke above resistance
            crossed_above = False
            cross_idx = 0
            for j in range(max(high_idx+1, i-24), i):
                if h[j] > recent_high and c[j-1] <= recent_high:
                    crossed_above = True
                    cross_idx = j
                    break
            
            if crossed_above and cross_idx > 0:
                # Price came back to retest the broken level
                retest_zone = recent_high * np.array([0.995, 1.005])
                
                # Look for retest: price near the broken level with indecision
                for j in range(cross_idx+3, i):
                    if (l[j] <= retest_zone[1] and l[j] >= retest_zone[0]):
                        # Indecision: small body candle
                        body = abs(c[j] - o[j])
                        range_c = h[j] - l[j]
                        if range_c > 0 and body/range_c < 0.4:  # Doji-like
                            # Wait for confirmation: next candle closes above
                            if j+1 < i and c[j+1] > o[j+1] and c[j+1] > c[j]:
                                # Entry signal
                                if c[i-1] > e200[i-1]:  # Trend filter: above EMA200
                                    pos=1; ep=c[i]
                                    break
        
        if not pos and cool>0: cool-=1
        cv.append(eq)
    
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

print("=== Role Reversal Strategy (تبادل الأدوار) ===")
for tp,sl in [(2,1),(3,1.5),(4,2)]:
    print(f"\n--- TP{tp}/SL{sl} ---")
    pr=[]; cr=[]
    for sym in LIQ:
        pd_=load(DP,sym); cd_=load(DC,sym)
        if pd_ is None or cd_ is None: continue
        r=role_reversal(*pd_,tp,sl)
        if r: r['sym']=sym; pr.append(r)
        r=role_reversal(*cd_,tp,sl)
        if r: r['sym']=sym; cr.append(r)
    
    for label,res in [('PREV',pr),('CUR',cr)]:
        if not res: continue
        tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
        tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
        wr=tw/tt*100 if tt>0 else 0
        dd=np.mean([x['dd'] for x in res]); gr=sum(1 for x in res if x['pnl']>0)
        print(f"  {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")

print("\nDone")
