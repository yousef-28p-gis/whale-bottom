#!/usr/bin/env python3
"""Multi-TF Clean A1/A2/A5 — 3-Year — FET 15m"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 1095; CAP = 1000

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

print('Fetching 3-year data...')
d1 = fetch('1d', DAYS); d4 = fetch('4h', DAYS); d1h = fetch('1h', DAYS); d15 = fetch('15m', DAYS)
print(f'  1d:{len(d1)} 4h:{len(d4)} 1h:{len(d1h)} 15m:{len(d15)}')

c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
n15=len(c15); idx15=d15.index

def align(hi, idx15):
    return pd.Series(hi.values, index=hi.index).reindex(idx15, method='ffill')

c1d=align(d1['close'],idx15).values; o1d=align(d1['open'],idx15).values
c4h=align(d4['close'],idx15).values; o4h=align(d4['open'],idx15).values
c1h=align(d1h['close'],idx15).values; o1h=align(d1h['open'],idx15).values
h4h=align(d4['high'],idx15).values; l4h=align(d4['low'],idx15).values

ema15_20 = pd.Series(c15).ewm(span=20, adjust=False).mean().values
ema15_50 = pd.Series(c15).ewm(span=50, adjust=False).mean().values
swing_low_15 = pd.Series(l15).rolling(10).min().values

def sim(le, tp, sl):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200, n15):
        if pos==1:
            if h15[i]>=ep*(1+tp/100):
                trades.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0
            elif c15[i]<=ep*(1-sl/100):
                pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve, eq

def rep(name, le, trades, curve, eq):
    if len(trades)<5: return
    w=[p for p in trades if p>0]; l=[p for p in trades if p<=0]
    wr=len(w)/len(trades)*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    rr=abs(aw/al) if al else 99
    dd=((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    dr=pd.Series(curve).pct_change().dropna()
    sh=(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
    ann=(eq/CAP)**(365/DAYS)-1
    ico='+' if eq>CAP else '-'
    print(f'{name:<30} {le.sum():>5}s {len(trades):>4d}t WR {wr:>5.1f}% R:R {rr:.2f}x DD {dd:>5.1f}% {ico}${eq-1000:>+8.0f} Sh{sh:>5.2f} Ann{ann*100:>+.0f}%')

print(f'\n{"="*85}')
print(f'Multi-TF Clean — 3-Year — FET 15m')
print(f'{"="*85}')

# A1: 3xGreen + SwingLow bounce
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i] and c4h[i]>o4h[i] and c1h[i]>o1h[i]:
        if c15[i]<=swing_low_15[i]*1.01 and c15[i]>c15[i-1] and c15[i]>o15[i]:
            le[i]=True
for tp,sl in [(3.0,2.0),(4.0,2.0),(5.0,2.5)]:
    tr,cv,eq=sim(le,tp,sl); rep(f'A1 3Green+SwLow TP{tp}SL{sl}',le,tr,cv,eq)

# A2: D+4h up + 1h pullback + 15m EMA20 bounce
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=ema15_20[i]*1.005 and c15[i]>ema15_20[i]*0.99 and c15[i]>c15[i-1]:
            le[i]=True
for tp,sl in [(3.0,2.0),(4.0,2.0),(5.0,2.5)]:
    tr,cv,eq=sim(le,tp,sl); rep(f'A2 D4hUp+PB+EMA20 TP{tp}SL{sl}',le,tr,cv,eq)

# A5: 1d up + 15m pullback to EMA50
le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i]:
        if l15[i]<=ema15_50[i]*0.995 and c15[i]>o15[i] and c15[i]>c15[i-1]:
            le[i]=True
for tp,sl in [(3.0,2.0),(4.0,2.0),(5.0,2.5)]:
    tr,cv,eq=sim(le,tp,sl); rep(f'A5 1dUp+PB@EMA50 TP{tp}SL{sl}',le,tr,cv,eq)

# Yearly breakdown for best
print(f'\n{"="*85}')
print('YEARLY BREAKDOWN — A2 D4hUp+PB+EMA20 TP4/SL2')
print(f'{"="*85}')

le=np.zeros(n15,bool)
for i in range(200,n15):
    if c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=ema15_20[i]*1.005 and c15[i]>ema15_20[i]*0.99 and c15[i]>c15[i-1]:
            le[i]=True

yearly={}
pos=0; ep=0
for i in range(200,n15):
    yr=idx15[i].year
    if yr not in yearly: yearly[yr]=[]
    if pos==1:
        if h15[i]>=ep*1.04:
            yearly[yr].append(3.8); pos=0  # TP4% - 0.2% comm
        elif c15[i]<=ep*0.98:
            pnl=(c15[i]/ep-1)*100-COMM*100; yearly[yr].append(pnl); pos=0
    if pos==0 and le[i]: pos=1; ep=c15[i]
if pos:
    pnl=(c15[-1]/ep-1)*100-COMM*100; yearly[list(yearly.keys())[-1]].append(pnl)

for yr in sorted(yearly.keys()):
    tr=yearly[yr]
    if not tr: continue
    w=[p for p in tr if p>0]; wr=len(w)/len(tr)*100
    print(f'  {yr}: {len(tr):>4d}t WR {wr:>5.1f}% Net {sum(tr):>+7.2f}%')
