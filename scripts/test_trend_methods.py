#!/usr/bin/env python3
"""Test Multi-TF Trend Detection Methods — FET 15m — 3 years"""
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

def ema(s,p): return s.ewm(span=p,adjust=False).mean()

print('Fetching...')
d1=fetch('1d',DAYS); d4=fetch('4h',DAYS); d1h=fetch('1h',DAYS); d15=fetch('15m',DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
n15=len(c15); idx15=d15.index

def align_shift(series, idx15):
    return pd.Series(series.shift(1).values, index=series.index).reindex(idx15, method='ffill').values

# Close prices (shifted)
c1d=align_shift(d1['close'],idx15); o1d=align_shift(d1['open'],idx15)
c4h=align_shift(d4['close'],idx15); o4h=align_shift(d4['open'],idx15)
c1h=align_shift(d1h['close'],idx15); o1h=align_shift(d1h['open'],idx15)

# EMAs
ema50_d=align_shift(ema(d1['close'],50),idx15)
ema200_d=align_shift(ema(d1['close'],200),idx15)
ema50_4h=align_shift(ema(d4['close'],50),idx15)
ema200_4h=align_shift(ema(d4['close'],200),idx15)
ema50_1h=align_shift(ema(d1h['close'],50),idx15)
ema200_1h=align_shift(ema(d1h['close'],200),idx15)

# SuperTrend on daily
def supertrend(h, l, c, factor=3.0, period=10):
    atr=pd.Series(h-l).rolling(period).mean().values
    hl2=(h+l)/2; n=len(c)
    up=hl2-factor*atr; dn=hl2+factor*atr
    trend=np.ones(n)
    for i in range(period,n):
        if c[i-1]>dn[i-1] if not np.isnan(dn[i-1]) else False: trend[i]=1
        elif c[i-1]<up[i-1] if not np.isnan(up[i-1]) else False: trend[i]=-1
        else: trend[i]=trend[i-1]
    return trend

st_d=align_shift(pd.Series(supertrend(d1['high'].values,d1['low'].values,d1['close'].values),index=d1.index),idx15)

# 15m indicators
ema20_15=pd.Series(c15).ewm(span=20,adjust=False).mean().values

# ENTRY: EMA20 bounce
def entry_ema20(c15, o15, ema20_15, n15):
    le=np.zeros(n15,bool)
    for i in range(200,n15):
        if c15[i]<=ema20_15[i]*1.005 and c15[i]>ema20_15[i]*0.99 and c15[i]>o15[i]:
            le[i]=True
    return le

base_entry = entry_ema20(c15, o15, ema20_15, n15)

def sim(le, tp, sl):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200,n15):
        if pos==1:
            if h15[i]>=ep*(1+tp/100):
                trades.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0
            elif c15[i]<=ep*(1-sl/100):
                trades.append((c15[i]/ep-1)*100-COMM*100); eq*=(1+((c15[i]/ep-1)*100-COMM*100)/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve, eq

# === TEST TREND METHODS ===
trend_methods = {}

# M1: Price > EMA50 on daily
trend_methods['M1 D:Price>EMA50'] = c1d > ema50_d

# M2: Price > EMA200 on daily
trend_methods['M2 D:Price>EMA200'] = c1d > ema200_d

# M3: EMA50 > EMA200 on daily (golden cross)
trend_methods['M3 D:EMA50>EMA200'] = ema50_d > ema200_d

# M4: Daily close > open
trend_methods['M4 D:Close>Open'] = c1d > o1d

# M5: SuperTrend daily bullish
trend_methods['M5 D:SuperTrend Up'] = st_d == 1

# M6: M3 + price > EMA50 on 4h + 1h (complete cascade)
trend_methods['M6 Cascade EMA50 all'] = (ema50_d > ema200_d) & (c4h > ema50_4h) & (c1h > ema50_1h)

# M7: Daily+4h+1h all green
trend_methods['M7 3xGreen'] = (c1d > o1d) & (c4h > o4h) & (c1h > o1h)

# M8: M3 + M7 combo
trend_methods['M8 Golden+3xGreen'] = (ema50_d > ema200_d) & (c4h > o4h) & (c1h > o1h)

# M9: Daily+4h SuperTrend both up
st_4h=align_shift(pd.Series(supertrend(d4['high'].values,d4['low'].values,d4['close'].values),index=d4.index),idx15)
trend_methods['M9 D+4h ST both up'] = (st_d==1) & (st_4h==1)

print(f'\n{"="*90}')
print(f'Multi-TF Trend Detection Methods — FET 15m — 3y')
print(f'{"="*90}')
print(f'{"Method":<30} {"Trend%":>7} {"Sigs":>6} {"Trades":>6} {"WR":>7} {"R:R":>6} {"DD":>7} {"Profit":>9}')
print('-'*85)

for name, trend_mask in trend_methods.items():
    le = base_entry.copy()
    le[~trend_mask] = False  # only enter when trend is bullish
    
    tr,cv,eq=sim(le,4.0,2.0)
    if len(tr)<5: continue
    w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    trend_pct=trend_mask.sum()/n15*100
    ico='+' if eq>CAP else '-'
    print(f'{name:<30} {trend_pct:>6.1f}% {le.sum():>6} {len(tr):>6} {wr:>6.1f}% {abs(aw/al) if al else 99:>5.2f}x {dd:>6.1f}% {ico}${eq-1000:>+8.0f}')
