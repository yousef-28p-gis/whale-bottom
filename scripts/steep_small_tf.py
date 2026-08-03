#!/usr/bin/env python3
"""Steep angle + pullback — 15m & 30m — FET/USDT"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(SYMBOL, tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def slope(c, period):
    y = c[-period:]; x = np.arange(period)
    return np.polyfit(x, y, 1)[0] / np.mean(y) * 100

def sim(c, h, l, o, le, se, tp, sl):
    n=len(c); w=100; trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(w, n):
        if pos==1:
            if h[i]>=ep*(1+tp/100):
                pnl=(ep*(1+tp/100)/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
            elif c[i]<=ep*(1-sl/100):
                pnl=(c[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
            elif se[i]:
                pnl=(c[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=-1; ep=c[i]
        elif pos==-1:
            if l[i]<=ep*(1-tp/100):
                pnl=(1-ep*(1-tp/100)/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
            elif c[i]>=ep*(1+sl/100):
                pnl=(1-c[i]/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
            elif le[i]:
                pnl=(1-c[i]/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=1; ep=c[i]
        if pos==0:
            if le[i]: pos=1; ep=c[i]
            elif se[i]: pos=-1; ep=c[i]
        curve.append(eq)
    if pos:
        pnl=((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def mets(tr, cv):
    if not tr or len(tr)<5: return None
    nt=len(tr); w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/nt*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return wr,abs(aw/al) if al else 99,dd,cv[-1],len(w),len(l),aw,al

for tf in ['15m', '30m']:
    print(f'\n{"="*75}')
    print(f'FET {tf} — Steep Angle + Pullback — Grid Search')
    print(f'{"="*75}')
    df = fetch(tf, DAYS)
    c=df['close'].values; h=df['high'].values; l=df['low'].values; o=df['open'].values
    n=len(c); w=100
    print(f'{len(df)} candles')
    
    # Pre-compute slopes & pullbacks
    lookback = 10  # shorter lookback for smaller TFs
    slopes = np.full(n, np.nan)
    pullbacks = np.full(n, np.nan)
    for i in range(50, n):
        slopes[i] = slope(c[i-lookback+1:i+1], lookback)
        peak5 = h[i-3:i+1].max()
        pullbacks[i] = (peak5 - c[i]) / peak5 * 100 if peak5 > 0 else 0
    
    angles = [0.15, 0.2, 0.25, 0.3, 0.4]
    pb_mins = [0.2, 0.3, 0.5]
    pb_maxs = [1.0, 1.5, 2.0, 2.5]
    tps = [0.8, 1.0, 1.5, 2.0]
    sls = [0.4, 0.5, 0.7, 1.0]
    
    tested = 0
    best = None
    for am in angles:
     for pb_min in pb_mins:
      for pb_max in pb_maxs:
       if pb_min >= pb_max: continue
       for tp in tps:
        for sl in sls:
         if sl >= tp: continue
        le=np.zeros(n,bool); se=np.zeros(n,bool)
        for i in range(w, n):
            if np.isnan(slopes[i]): continue
            steep = any(not np.isnan(slopes[j]) and slopes[j]>am for j in range(max(0,i-5),i+1))
            steep_down = any(not np.isnan(slopes[j]) and slopes[j]<-am for j in range(max(0,i-5),i+1))
            if steep and pullbacks[i]>pb_min and pullbacks[i]<pb_max and c[i]>h[i-1] and c[i]>o[i]:
                le[i]=True
            pb_up = (c[i]-l[max(0,i-3):i+1].min())/l[max(0,i-3):i+1].min()*100
            if steep_down and pb_up>pb_min and pb_up<pb_max and c[i]<l[i-1] and c[i]<o[i]:
                se[i]=True
        
        if le.sum()+se.sum() < 5: continue
        tested += 1
        tr,cv = sim(c,h,l,o,le,se,tp,sl)
        mr = mets(tr,cv)
        if not mr: continue
        wr,rr,dd,eq,nw,nl,aw,al = mr
        sc = eq/CAP * (wr/100)
        if best is None or (wr>35 and eq>CAP and sc>best.get('sc',0)):
            best = {'sc':sc,'wr':wr,'rr':rr,'dd':dd,'eq':eq,'n':nw+nl,'a':am,'pb_min':pb_min,'pb_max':pb_max,'tp':tp,'sl':sl,'sig':le.sum()+se.sum()}
    
    if best:
        print(f'  Best: angle>{best["a"]:.2f}% PB {best["pb_min"]:.1f}-{best["pb_max"]:.1f}% TP{best["tp"]} SL{best["sl"]}')
        print(f'  {best["n"]}t ({best["sig"]}s) WR {best["wr"]:.1f}% R:R {best["rr"]:.2f}x DD {best["dd"]:.1f}% ${best["eq"]-1000:+.0f}')
    print(f'  {tested} combos tested')

# Top 10 for best TF
print(f'\n{"="*75}')
print('TOP 10 — 15m (all profitable configs)')
print(f'{"="*75}')
df=fetch('15m',DAYS)
c=df['close'].values; h=df['high'].values; l=df['low'].values; o=df['open'].values; n=len(c)
slopes=np.full(n,np.nan); pullbacks=np.full(n,np.nan)
for i in range(50,n): 
    slopes[i]=slope(c[i-9:i+1],10)
    pullbacks[i]=(h[i-3:i+1].max()-c[i])/h[i-3:i+1].max()*100 if h[i-3:i+1].max()>0 else 0

results=[]
for am in [0.2,0.25,0.3,0.35]:
 for pb_min in [0.2,0.3,0.5]:
  for pb_max in [1.0,1.5,2.0]:
   if pb_min >= pb_max: continue
   for tp in [1.0,1.5,2.0]:
    for sl in [0.5,0.7,1.0]:
     if sl >= tp: continue
    le=np.zeros(n,bool); se=np.zeros(n,bool)
    for i in range(100,n):
        if np.isnan(slopes[i]): continue
        steep=any(not np.isnan(slopes[j]) and slopes[j]>am for j in range(max(0,i-5),i+1))
        steep_down=any(not np.isnan(slopes[j]) and slopes[j]<-am for j in range(max(0,i-5),i+1))
        if steep and pullbacks[i]>pb_min and pullbacks[i]<pb_max and c[i]>h[i-1] and c[i]>o[i]: le[i]=True
        pb_up=(c[i]-l[max(0,i-3):i+1].min())/l[max(0,i-3):i+1].min()*100
        if steep_down and pb_up>pb_min and pb_up<pb_max and c[i]<l[i-1] and c[i]<o[i]: se[i]=True
    if le.sum()+se.sum()<5: continue
    tr,cv=sim(c,h,l,o,le,se,tp,sl)
    mr=mets(tr,cv)
    if mr: 
        wr,rr,dd,eq,nw,nl,aw,al=mr
        if eq>CAP: results.append((eq,wr,dd,rr,nw+nl,le.sum()+se.sum(),aw,al,f'a>{am} PB{pb_min}-{pb_max} TP{tp} SL{sl}'))

for i,x in enumerate(sorted(results, key=lambda x: x[1], reverse=True)[:10]):
    eq,wr,dd,rr,n,sg,aw,al,nm=x
    print(f'{i+1:>2}. {nm:<35} {n:>4d}t WR {wr:>5.1f}% R:R {rr:.2f}x DD {dd:>5.1f}% ${eq-1000:>+7.0f}')
