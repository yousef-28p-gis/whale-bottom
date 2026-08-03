#!/usr/bin/env python3
"""Whale+SSL — FET/USDT 1m — 7d — مصحح shift+close"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=7)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('FET/USDT','1m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c)
idx=df.index

# SSL v2 — مصحح
period=10
sma_h=pd.Series(h).rolling(period).mean().values
sma_l=pd.Series(l_).rolling(period).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(period,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

# Whale — shift(1) لا Look-ahead
LB=50
ln=pd.Series(l_).shift(1).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# Entry: مصحح shift(1) + close-only
le=np.zeros(n,bool)
for i in range(200,n):
    if i>0 and ssl_c[i-1]==1 and wp_up[i-1] and wp[i-1]>wp[max(0,i-3)]*2 and wp[i-1]>0:
        le[i]=True

p_change=(c[-1]-c[0])/c[0]*100

print(f'FET 1m | 7d | {n} candles | {le.sum()} signals | السعر Δ {p_change:+.1f}%')
print(f'{"TP/SL":>10} {"T":>4} {"WR":>6} {"DD":>6} {"Eq":>9} {"W/L":>8}')
print('-'*50)

for tp,sl in [(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+tp/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
            elif c[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
        if not pos and cool==0 and le[i]: pos=1; ep=o[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='✅' if eq>CAP else '❌'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>3} {wr:>5.1f}% {dd:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

# Also try whale only on FET
print(f'\n🐋 Whale فقط (بدون SSL):')
le_w=np.zeros(n,bool)
for i in range(200,n):
    if i>0 and wp_up[i-1] and wp[i-1]>wp[max(0,i-3)]*2 and wp[i-1]>0:
        le_w[i]=True
print(f'   {le_w.sum()} signals')

for tp,sl in [(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+tp/100): pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
            elif c[i]<=ep*(1-sl/100): pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
        if not pos and cool==0 and le_w[i]: pos=1; ep=o[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100; ico='✅' if eq>CAP else '❌'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>3} {wr:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

print('\n✅ Done')
