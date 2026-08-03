#!/usr/bin/env python3
"""FET 3m — Whale+SSL — أول حوت بعد SSL أزرق"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

ex=ccxt.binance({'timeout':15000})
since=ex.parse8601((datetime.utcnow()-timedelta(days=14)).isoformat())
all_c=[]
while True:
    batch=ex.fetch_ohlcv('FET/USDT','3m',since=since,limit=1000)
    if not batch: break
    all_c.extend(batch)
    since=batch[-1][0]+1
    if len(batch)<1000: break

df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
df['ts']=pd.to_datetime(df['ts'],unit='ms')
df.set_index('ts',inplace=True); df.sort_index(inplace=True)

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c); idx=df.index

# SSL v2
p=10
sma_h=pd.Series(h).rolling(p).mean().values
sma_l=pd.Series(l_).rolling(p).mean().values
ssl=np.full(n,np.nan); ssl_c=np.zeros(n,int)
for i in range(p,n):
    if h[i-1]>sma_h[i-1]: ssl[i]=sma_l[i]; ssl_c[i]=1
    else: ssl[i]=sma_h[i]; ssl_c[i]=-1

# Whale
LB=50
ln=pd.Series(l_).shift(1).rolling(LB).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(LB).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)

# SSL flips
ssl_flip_blue=np.zeros(n,bool)
for i in range(1,n):
    if ssl_c[i]==1 and ssl_c[i-1]==-1:
        ssl_flip_blue[i]=True

# دخول: أول حوت بعد SSL يتحول أزرق
le=np.zeros(n,bool)
waiting=False
for i in range(200,n):
    if ssl_flip_blue[i]:
        waiting=True
    if waiting and wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0:
        le[i]=True; waiting=False

p_change=(c[-1]-c[0])/c[0]*100
print(f'FET 3m | 14d | {n} candles | Δ {p_change:+.1f}%')
print(f'SSL flips blue: {ssl_flip_blue.sum()} | Entries: {le.sum()}')
print(f'{"TP/SL":>10} {"T":>4} {"WR":>6} {"DD":>6} {"Eq":>9} {"W/L":>8}')
print('-'*50)

for tp,sl in [(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+tp/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=10
            elif c[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=10
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

# Details
print(f'\nتفاصيل (TP5/SL2.5):')
trades=[]; pos=0; ep=0; ei=0
for i in range(200,n):
    if pos:
        if c[i]>=ep*(1+5/100): trades.append((ei,i,ep,c[i],'TP')); pos=0
        elif c[i]<=ep*(1-2.5/100): trades.append((ei,i,ep,c[i],'SL')); pos=0
    if not pos and le[i]: pos=1; ep=o[i]; ei=i
for i,(ei,xi,ep,xp,tt) in enumerate(trades):
    pnl=(xp/ep-1)*100-COMM*100
    print(f'  {i+1}. {tt} | {idx[ei]} → {idx[xi]} | ${ep:.5f}→${xp:.5f} | {pnl:+.2f}% | {(xi-ei)*3}min')

print(f'\n✅ Done | {len(trades)} trades')
