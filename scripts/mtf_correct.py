#!/usr/bin/env python3
"""Multi-TF — CORRECT: TP=HIGH, SL=CLOSE, Daily trend shifted — FET 15m 3y"""
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

print('Fetching...')
d1=fetch('1d',DAYS); d4=fetch('4h',DAYS); d1h=fetch('1h',DAYS); d15=fetch('15m',DAYS)
c15=d15['close'].values; h15=d15['high'].values; l15=d15['low'].values; o15=d15['open'].values
v15=d15['volume'].values; n15=len(c15); idx15=d15.index
print(f'1d:{len(d1)} 4h:{len(d4)} 1h:{len(d1h)} 15m:{len(d15)}')

# CORRECT: shift(1) daily/4h/1h to yesterday's data
def align_shifted(series, idx15):
    return pd.Series(series.shift(1).values, index=series.index).reindex(idx15, method='ffill').values

c1d = align_shifted(d1['close'], idx15)
o1d = align_shifted(d1['open'], idx15)
c4h = align_shifted(d4['close'], idx15)
o4h = align_shifted(d4['open'], idx15)
c1h = align_shifted(d1h['close'], idx15)
o1h = align_shifted(d1h['open'], idx15)

# 15m indicators
ema20_15 = pd.Series(c15).ewm(span=20, adjust=False).mean().values
ema50_15 = pd.Series(c15).ewm(span=50, adjust=False).mean().values
swing_low_15 = pd.Series(l15).rolling(10).min().values

daily_bullish = c1d > o1d
print(f'Daily bullish: {daily_bullish.sum()/n15*100:.1f}%')

def sim(le, tp, sl):
    trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(200, n15):
        if pos==1:
            if h15[i] >= ep*(1+tp/100):  # TP = HIGH (limit order)
                trades.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0
            elif c15[i] <= ep*(1-sl/100):  # SL = CLOSE
                pnl=(c15[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
        if pos==0 and le[i]: pos=1; ep=c15[i]
        curve.append(eq)
    if pos:
        pnl=(c15[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve, eq

def rep(name, le, tp, sl):
    tr,cv,eq=sim(le,tp,sl)
    if len(tr)<5: return None
    w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/len(tr)*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    dr=pd.Series(cv).pct_change().dropna()
    sh=(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
    ann=(eq/CAP)**(365/DAYS)-1
    tps=sum(1 for p in tr if p>0); sls=sum(1 for p in tr if p<=0)
    ico='+' if eq>CAP else '-'
    print(f'{name:<40} {le.sum():>5}s {len(tr):>4d}t WR{wr:>5.1f}% R:R{abs(aw/al) if al else 99:>5.2f}x DD{dd:>5.1f}% {ico}${eq-1000:>+8.0f} TP{tps} SL{sls} Sh{sh:.2f}')
    return wr, dd, eq, len(tr)

print(f'\n{"="*90}')
print(f'CORRECT LOGIC: TP=HIGH SL=CLOSE DailyShift — FET 15m 3y')
print(f'{"="*90}')

# A1: Daily green + 15m pullback to swing low + green bounce
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not daily_bullish[i]: continue
    if c15[i]<=swing_low_15[i]*1.01 and c15[i]>o15[i] and c15[i]>c15[i-1]:
        le[i]=True
rep('A1 DailyUp + SwingLow bounce TP4/SL2', le, 4.0, 2.0)
rep('A1 DailyUp + SwingLow bounce TP3/SL2', le, 3.0, 2.0)

# A2: Daily green + 15m touch EMA20 + green bounce
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not daily_bullish[i]: continue
    if c15[i]<=ema20_15[i]*1.005 and c15[i]>ema20_15[i]*0.99 and c15[i]>o15[i]:
        le[i]=True
rep('A2 DailyUp + EMA20 bounce TP4/SL2', le, 4.0, 2.0)
rep('A2 DailyUp + EMA20 bounce TP3/SL2', le, 3.0, 2.0)

# A5: Daily green + 15m deep pullback to EMA50 + reversal
le=np.zeros(n15,bool)
for i in range(200,n15):
    if not daily_bullish[i]: continue
    if l15[i]<=ema50_15[i]*0.995 and c15[i]>o15[i] and c15[i]>c15[i-1]:
        le[i]=True
rep('A5 DailyUp + DeepPB@EMA50 TP4/SL2', le, 4.0, 2.0)
rep('A5 DailyUp + DeepPB@EMA50 TP3/SL2', le, 3.0, 2.0)

# A6: Daily+4h green + 15m EMA20 bounce (original A2)
le=np.zeros(n15,bool)
daily_and_4h = daily_bullish & (c4h > o4h)
for i in range(200,n15):
    if not daily_and_4h[i]: continue
    if c15[i]<=ema20_15[i]*1.005 and c15[i]>ema20_15[i]*0.99 and c15[i]>o15[i]:
        le[i]=True
rep('A6 Daily+4hUp + EMA20 TP4/SL2', le, 4.0, 2.0)
