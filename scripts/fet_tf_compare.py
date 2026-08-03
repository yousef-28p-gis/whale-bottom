#!/usr/bin/env python3
"""FET — قارن كل الفريمات — 3m 5m 15m — 14d"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000

def test_tf(symbol, tf, days=14):
    ex=ccxt.binance({'timeout':15000})
    since=ex.parse8601((datetime.utcnow()-timedelta(days=days)).isoformat())
    all_c=[]
    while True:
        batch=ex.fetch_ohlcv(symbol,tf,since=since,limit=1000)
        if not batch: break
        all_c.extend(batch)
        since=batch[-1][0]+1
        if len(batch)<1000: break
    df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','volume'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms')
    df.set_index('ts',inplace=True); df.sort_index(inplace=True)
    
    c=df['close'].values; h=df['high'].values
    l_=df['low'].values; o=df['open'].values; n=len(c); idx=df.index
    
    # SSL
    p=10
    sma_h=pd.Series(h).rolling(p).mean().values
    sma_l=pd.Series(l_).rolling(p).mean().values
    ssl_c=np.zeros(n,int)
    for i in range(p,n):
        if h[i-1]>sma_h[i-1]: ssl_c[i]=1
        else: ssl_c[i]=-1
    
    ssl_flip_blue=np.zeros(n,bool)
    for i in range(1,n):
        if ssl_c[i]==1 and ssl_c[i-1]==-1: ssl_flip_blue[i]=True
    
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
    
    # Entry
    le=np.zeros(n,bool); waiting=False
    for i in range(200,n):
        if ssl_flip_blue[i]: waiting=True
        if waiting and wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0:
            le[i]=True; waiting=False
    
    ch=(c[-1]-c[0])/c[0]*100
    
    # Test TP3/SL1.5
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if c[i]>=ep*(1+3/100): pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=10
            elif c[i]<=ep*(1-1.5/100): pnl=(c[i]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=10
        if not pos and cool==0 and le[i]: pos=1; ep=o[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos: pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
    
    w=[p for p in t if p>0]; lo=[p for p in t if p<=0]
    wr=len(w)/len(t)*100 if t else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min() if t else 0
    
    return {'tf':tf,'n':n,'ch':ch,'ssl_flips':ssl_flip_blue.sum(),'entries':le.sum(),
            'trades':len(t),'wr':wr,'dd':dd,'eq':eq,'w':len(w),'l':len(lo)}

print('FET — مقارنة الفريمات — 14d — TP3/SL1.5')
print(f'{"فريم":>6} {"شموع":>6} {"Δ%":>6} {"SSL↔":>6} {"إشارات":>6} {"صفقات":>5} {"WR":>6} {"سحب":>6} {"Eq":>8}')
print('-'*65)

for tf in ['1m','3m','5m','15m']:
    r=test_tf('FET/USDT',tf)
    ico='✅' if r['eq']>CAP else '❌'
    print(f'{r["tf"]:>6} {r["n"]:>6} {r["ch"]:>+5.1f}% {r["ssl_flips"]:>6} {r["entries"]:>6} {r["trades"]:>5} {r["wr"]:>5.1f}% {r["dd"]:>5.1f}% {ico}${r["eq"]-CAP:>+7.1f}')

print('\n✅ Done')
