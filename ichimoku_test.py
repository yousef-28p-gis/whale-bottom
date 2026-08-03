#!/usr/bin/env python3
"""Ichimoku Kinko Hyo Strategy — Tenkan/Kijun Cross + Cloud Filter"""
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

def ichimoku(c,h,l,o,ts,tp,sl):
    n=len(c)
    # Tenkan = (9-period highest high + 9-period lowest low) / 2
    h9=pd.Series(h).rolling(9).max().values
    l9=pd.Series(l).rolling(9).min().values
    tenkan=(h9+l9)/2
    
    # Kijun = (26-period highest high + 26-period lowest low) / 2
    h26=pd.Series(h).rolling(26).max().values
    l26=pd.Series(l).rolling(26).min().values
    kijun=(h26+l26)/2
    
    # Senkou A = (Tenkan + Kijun) / 2 shifted 26 forward
    senkou_a=np.full(n,np.nan)
    for i in range(26,n-26):
        senkou_a[i+26]=(tenkan[i]+kijun[i])/2
    
    # Senkou B = (52-period high+low)/2 shifted 26 forward
    h52=pd.Series(h).rolling(52).max().values
    l52=pd.Series(l).rolling(52).min().values
    senkou_b=np.full(n,np.nan)
    for i in range(52,n-26):
        senkou_b[i+26]=(h52[i]+l52[i])/2
    
    # Cloud (use shifted values at current position)
    # At position i, the cloud is between senkou_a[i] and senkou_b[i]
    # Price above cloud = c[i] > max(senkou_a[i], senkou_b[i])
    # Price below cloud = c[i] < min(senkou_a[i], senkou_b[i])
    # Price in cloud = between them
    
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; side=0
    
    for i in range(100,n):
        if np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]): continue
        
        cloud_top=max(senkou_a[i], senkou_b[i])
        cloud_bot=min(senkou_a[i], senkou_b[i])
        in_cloud=c[i]>=cloud_bot and c[i]<=cloud_top
        above_cloud=c[i]>cloud_top
        below_cloud=c[i]<cloud_bot
        
        if pos:
            # Exit: reverse cross or TP/SL
            if side==1:
                if tenkan[i]<kijun[i] and tenkan[i-1]>=kijun[i-1]:  # Death cross
                    pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-sl/100):
                    pnl=max((c[i]/ep-1)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
            else:
                if tenkan[i]>kijun[i] and tenkan[i-1]<=kijun[i-1]:  # Golden cross
                    pnl=(1-c[i]/ep)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif l[i]<=ep*(1-tp/100):
                    pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
                elif h[i]>=ep*(1+sl/100):
                    pnl=max((1-c[i]/ep)*100-COMM*100,-sl*MAX_SLIPPAGE-COMM*100)
                    t.append(pnl); eq*=(1+pnl/100); pos=0; cool=COOLDOWN
        
        if not pos and cool==0:
            if not in_cloud:  # Don't trade in cloud
                if above_cloud:
                    # Golden cross: Tenkan crosses above Kijun
                    if tenkan[i]>kijun[i] and tenkan[i-1]<=kijun[i-1]:
                        pos=1; ep=c[i]; side=1
                elif below_cloud:
                    # Death cross
                    if tenkan[i]<kijun[i] and tenkan[i-1]>=kijun[i-1]:
                        pos=1; ep=c[i]; side=-1
        
        if not pos and cool>0: cool-=1
        cv.append(eq)
    
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100 if side==1 else (1-c[-1]/ep)*100-COMM*100
        t.append(pnl); eq*=(1+pnl/100)
    
    if len(t)<5: return None
    w=sum(1 for p in t if p>0)
    return {'t':len(t),'wr':w/len(t)*100,'dd':((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min(),'pnl':eq-CAP,'w':w,'l':len(t)-w}

print("=== Ichimoku Strategy — Tenkan/Kijun Cross + Cloud ===")
for tp,sl in [(3,1.5),(4,2),(5,2.5)]:
    print(f"\n--- TP{tp}/SL{sl} ---")
    pr=[]; cr=[]
    for sym in LIQ:
        pd_=load(DP,sym); cd_=load(DC,sym)
        if pd_ is None or cd_ is None: continue
        r=ichimoku(*pd_,tp,sl)
        if r: r['sym']=sym; pr.append(r)
        r=ichimoku(*cd_,tp,sl)
        if r: r['sym']=sym; cr.append(r)
    
    for label,res in [('PREV',pr),('CUR',cr)]:
        if not res: continue
        tt=sum(x['t'] for x in res); tw=sum(x['w'] for x in res)
        tl=sum(x['l'] for x in res); tp_=sum(x['pnl'] for x in res)
        wr=tw/tt*100 if tt>0 else 0
        dd=np.mean([x['dd'] for x in res]); gr=sum(1 for x in res if x['pnl']>0)
        print(f"  {label}: {len(res)}Ⓜ️ {tt}T 🟢{tw} 🔴{tl} WR={wr:.1f}% DD={dd:.1f}% ${tp_:+,.0f} g={gr}")

print("\nDone")
