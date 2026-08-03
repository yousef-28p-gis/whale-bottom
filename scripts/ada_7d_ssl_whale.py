#!/usr/bin/env python3
"""SSL v2 + Whale — ADA 1m — 7 days backtest"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=7)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('ADA/USDT','1m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c)

# SSL v2 — خط واحد
period=10
sma_h=pd.Series(h).rolling(period).mean().values
sma_l=pd.Series(l_).rolling(period).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(period,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

# Whale
LB=50
ln=pd.Series(l_).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# Entry
le=np.zeros(n,bool)
for i in range(200,n):
    if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
        le[i]=True

p_change=(c[-1]-c[0])/c[0]*100

print(f'ADA 1m | 7d | {n} candles | {le.sum()} signals | Δprice {p_change:+.1f}%')
print(f'{"TP/SL":>10} {"Trades":>5} {"WR":>6} {"R:R":>5} {"DD":>6} {"Equity":>9} {"W/L":>8}')
print('-'*55)

for tp,sl in [(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=12
            elif l_[i]<=ep*(1-sl/100):
                t.append(-sl-COMM*100); eq*=(1+(-sl-COMM*100)/100); pos=0; cool=12
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    ico='✅' if eq>CAP else '❌'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>5} {wr:>5.1f}% {aw/(al+0.001):>4.2f}x {dd:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

# Compare: without SSL (whale only)
le_w=np.zeros(n,bool)
for i in range(200,n):
    if wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
        le_w[i]=True

print(f'\n🐋 بدون SSL ({le_w.sum()} signals):')
print(f'{"TP/SL":>10} {"Trades":>5} {"WR":>6} {"Equity":>9} {"W/L":>8}')
print('-'*40)
for tp,sl in [(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100): t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=12
            elif l_[i]<=ep*(1-sl/100): t.append(-sl-COMM*100); eq*=(1+(-sl-COMM*100)/100); pos=0; cool=12
        if not pos and cool==0 and le_w[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    if len(t)<3: continue
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100
    ico='✅' if eq>CAP else '❌'
    print(f'{tp:.1f}%/{sl:.1f}%   {len(t):>5} {wr:>5.1f}% {ico}${eq-CAP:>+8.1f} {len(w)}W/{len(lo)}L')

print('\n✅ Done')
