#!/usr/bin/env python3
"""Multi-TF Clean — FIXED — Close-only + no look-ahead + fixed sizing"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 1095; CAP = 1000; FIXED_SIZE = 500

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

print('Fetching 3-year...')
d1 = fetch('1d', DAYS); d4 = fetch('4h', DAYS); d1h = fetch('1h', DAYS); d15 = fetch('15m', DAYS)
print(f'1d:{len(d1)} 4h:{len(d4)} 1h:{len(d1h)} 15m:{len(d15)}')

c15=d15['close'].values; n15=len(c15); idx15=d15.index

# FIX: shift(1) all higher TF data to prevent look-ahead
def align_shifted(higher, idx15):
    shifted = higher.shift(1)  # use PREVIOUS bar's close, not current
    return pd.Series(shifted.values, index=shifted.index).reindex(idx15, method='ffill')

c1d=align_shifted(d1['close'],idx15).values; o1d=align_shifted(d1['open'],idx15).values
c4h=align_shifted(d4['close'],idx15).values; o4h=align_shifted(d4['open'],idx15).values

ema15_20 = pd.Series(c15).ewm(span=20, adjust=False).mean().values

# FIX: Close-only simulation (no high/low for TP)
def sim_close_only(le, tp_pct, sl_pct):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200, n15):
        if pos==1:
            if c15[i] >= ep*(1+tp_pct/100):
                pnl_pct = tp_pct - COMM*100
                trades.append(pnl_pct); eq*=(1+pnl_pct/100); pos=0
            elif c15[i] <= ep*(1-sl_pct/100):
                pnl_pct = (c15[i]/ep-1)*100-COMM*100
                trades.append(pnl_pct); eq*=(1+pnl_pct/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl_pct = (c15[-1]/ep-1)*100-COMM*100
        trades.append(pnl_pct); eq*=(1+pnl_pct/100); curve.append(eq)
    return trades, curve, eq

def report(name, le, trades, curve, eq):
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
    print(f'{name:<35} {le.sum():>5}s {len(trades):>4d}t WR {wr:>5.1f}% R:R {rr:.2f}x DD {dd:>5.1f}% {ico}${eq-1000:>+9.0f} Sh{sh:>5.2f} Ann{ann*100:>+5.0f}%')

# A1: 3xGreen + SwingLow
le=np.zeros(n15,bool)
swing_low_15 = pd.Series(d15['low']).rolling(10).min().values
for i in range(200,n15):
    if not np.isnan(c1d[i]) and c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=swing_low_15[i]*1.01 and c15[i]>d15['open'].iloc[i] and c15[i]>c15[i-1]:
            le[i]=True
tr,cv,eq=sim_close_only(le,3.0,2.0); report('A1 3Green+SwLow CLOSE TP3/SL2',le,tr,cv,eq)
tr,cv,eq=sim_close_only(le,4.0,2.0); report('A1 3Green+SwLow CLOSE TP4/SL2',le,tr,cv,eq)

# A2: D+4h up + EMA20 bounce
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not np.isnan(c1d[i]) and c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=ema15_20[i]*1.005 and c15[i]>ema15_20[i]*0.99 and c15[i]>d15['open'].iloc[i]:
            le[i]=True
tr,cv,eq=sim_close_only(le,3.0,2.0); r2a=report('A2 D4h+EMA20 CLOSE TP3/SL2',le,tr,cv,eq)
tr,cv,eq=sim_close_only(le,4.0,2.0); r2b=report('A2 D4h+EMA20 CLOSE TP4/SL2',le,tr,cv,eq)

# A5: 1d only + EMA50 pullback
ema15_50 = pd.Series(c15).ewm(span=50, adjust=False).mean().values
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not np.isnan(c1d[i]) and c1d[i]>o1d[i]:
        if d15['low'].iloc[i]<=ema15_50[i]*0.995 and c15[i]>d15['open'].iloc[i] and c15[i]>c15[i-1]:
            le[i]=True
tr,cv,eq=sim_close_only(le,3.0,2.0); report('A5 1d+EMA50 CLOSE TP3/SL2',le,tr,cv,eq)
tr,cv,eq=sim_close_only(le,4.0,2.0); report('A5 1d+EMA50 CLOSE TP4/SL2',le,tr,cv,eq)

# Yearly for best
print(f'\n--- Yearly A2 CLOSE TP4/SL2 ---')
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not np.isnan(c1d[i]) and c1d[i]>o1d[i] and c4h[i]>o4h[i]:
        if c15[i]<=ema15_20[i]*1.005 and c15[i]>ema15_20[i]*0.99 and c15[i]>d15['open'].iloc[i]:
            le[i]=True

yearly={}; pos=0; ep=0
for i in range(200,n15):
    yr=idx15[i].year
    if yr not in yearly: yearly[yr]=[]
    if pos==1:
        if c15[i]>=ep*1.04:
            yearly[yr].append(3.8); pos=0
        elif c15[i]<=ep*0.98:
            yearly[yr].append((c15[i]/ep-1)*100-COMM*100); pos=0
    if pos==0 and le[i]: pos=1; ep=c15[i]

for yr in sorted(yearly.keys()):
    tr=yearly[yr]
    if tr: 
        w=[p for p in tr if p>0]
        print(f'  {yr}: {len(tr):>4d}t WR {len(w)/len(tr)*100:>5.1f}% Net {sum(tr):>+7.1f}%')
